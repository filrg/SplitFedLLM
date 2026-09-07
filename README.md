# SplitFedLLM
## Setup environment
When executing on DAI, access to a virtual environment is required.
```commandline
source sl/bin/activate
```
## Configuration
Application configuration is in the `config.yaml` file:
```yaml
name: SplitFedLLM
server:
  global-round: 1
  clients:
    - 1
    - 1
  cut-layers: 4
  model-name: Bert # GPT2/Llama/Bert
  data-name: EMOTION # EMOTION/GSM8K
  model:
    GPT2:
      n_block: 12
    Llama:
      n_block: 12
    Bert:
      n_block: 12
  parameters:
    load: True
    save: True
  validation: True
  data-distribution:
    non-iid: False
    num-sample: 500
    num-label: 10
    dirichlet:
      alpha: 1
    refresh-each-round: True
  random-seed: 1

rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

log_path: .
debug_mode: True

learning:
  learning-rate: 0.00001
  weight-decay: 0.01
  batch-size: 2
  control-count: 1
  clip-grad-norm: 0.0

fine-tune:
  name: LoRA
  LoRA:
    r: 8
    alpha: 16
```
## How to Run
### Server
```commandline
python server.py
```
### Client
```commandline
python client.py --layer_id 1
```
Where:
- `--layer_id` is the index of client's layer, start from 1

## U-shape runtime (default)

`server.architecture: u-shape` keeps embeddings/early blocks **and the output
head/loss** on the data owner. Remote workers run only the transformer body.
Token IDs, labels, logits, predictions and per-example losses are never included
in U-shape messages. Activations, hidden-state gradients, model updates and padding
metadata still leave the device; this is not a formal guarantee against inference
attacks on those values.

```text
Owner/front --hidden--> Worker/body --hidden--> Owner/tail + local loss
Owner/front <--grad---- Worker/body <--grad---- Owner/tail backward
```

`cut-layers` is the number of front blocks; `tail-blocks` is the number of final
blocks also kept on the owner. With `tail-blocks: 0`, the head and final
normalization (or BERT pooler/classifier) still remain local. There must be at
least one remote body block. LoRA and full fine-tuning work with all three models.
The existing models use untied embedding/head weights; the partition factory
does not add support for externally supplied architectures with tied weights.

Example with four owners and two workers:

```yaml
server:
  architecture: u-shape
  clients: [4, 2]
  cut-layers: 4
  tail-blocks: 0
learning:
  u-shape:
    max-inflight: 2
    microbatches-per-window: 8
    worker-max-inflight: 8
    timeout-seconds: 120
```

Run `python server.py`, then start four owner processes with
`python client.py --layer_id 1` and two worker processes with
`python client.py --layer_id 2`. Each process uses the configured broker;
`--device cpu` or `--device cuda` selects its compute device. The coordinator
assigns owners evenly to workers and pins each group's routes for the round.
There must be at least as many owners as workers.

`max-inflight` bounds live microbatches per owner. `worker-max-inflight` bounds
live body graphs across the group and must be at least its number of owners.
The effective per-owner credit is the smaller of `max-inflight` and
`floor(worker-max-inflight / owners_in_group)`. `microbatches-per-window` is a
per-owner update quota, independent of the credit limit. A released slot is
refilled immediately within the same window; backwards and tail work are serviced
before new forwards. Different owners/workers can overlap computation without
concurrent writes to a GPU's parameter gradients.

All group members keep parameters unchanged until every owner has drained its
window. The worker steps and sends COMMIT; owners step front+tail together before
entering the next window. The objective is the **mean of per-microbatch mean
losses across the group**, scaled once at the local loss. This is not a global
token-weighted mean when batch sizes/token counts differ. Positive gradient
clipping applies separately to the owner partition and worker partition. Empty
owners and partial final windows participate in the barrier without extra steps.

Each group trains independently. At round end, `src/UShapeServer.py` FedAvgs
owner and body partitions using processed sample counts. `parameters.save` controls
disk persistence; aggregation and redistribution occur even when saving is off.
The original full-model checkpoint key names are preserved, and checkpoints are
written via a temporary file and atomic replacement. Validation, when enabled,
runs locally at owners before FedAvg and reports only local mean loss; it does
not perform the legacy centralized generation/accuracy evaluation.

The protocol uses session/window IDs, dedicated per-node queues, manual ACKs,
publisher confirms in RpcClient, and duplicate suppression within a live process.
Timeout, nonfinite tensors/loss or peer failure aborts the round. It does **not**
resume a lost autograd graph or provide durable exactly-once optimizer updates:
restart all participants from the last completed round checkpoint. Old U-shape
queues are removed by the existing testbed startup cleanup.

Implementation: `src/model/u_shape.py` (partitions/checkpoint/LoRA),
`src/fine_tune/u_shape.py` (common scheduler), `src/UShapeServer.py` (coordinator),
and the U-shape branch in `src/RpcClient.py`. See
[the architecture design](docs/u_shape_design.md) for rationale and planned
optimizations. Direct tensor transport, adaptive routing and cross-owner tensor
batching remain profiling-driven extensions; the current data plane is RabbitMQ.

Run the offline regression/integration tests:

```commandline
python -m unittest discover -s tests -v
```

Tests exercise actual small BERT/GPT-2/Llama partitions, monolithic gradient and
update equivalence, dropout, graph release, LoRA merge, multiple owners, uneven
windows, reordering/duplicates, local validation, and two-round RpcClient/FedAvg
integration using an in-memory broker. No model download is needed for tests.
Real-broker/GPU throughput must be measured on the target testbed.

## Legacy two-stage scheduler (`server.architecture: split`)

This opt-in compatibility mode sends labels to the last stage. Use U-shape for
local-label training.

`src/fine_tune/scheduler.py` implements the first/last-stage training loops,
RabbitMQ transport, backpressure, optimizer steps and round completion for all
three models. `RpcClient` creates it with `create_scheduler(model_name, ...)`.
The old `Ft_Bert`, `Ft_GPT2` and `Ft_Llama` constructors and training methods
delegate to this implementation.

The first stage stores the original output tensor and autograd graph under its
`data_id`. Only the transmitted copy is detached. A returning gradient calls
`output.backward(gradient)` on that graph, without repeating forward (including
dropout). Each graph is released after backward, and NOTIFY is sent only after
all batches have completed backward and their optimizer updates.

`learning.control-count` must be a positive integer. It limits a window of live
first-stage graphs. With `1`, each batch performs one forward, one backward and
one optimizer step. With values above `1`, the first stage sends up to that many
batches, drains their gradients in any arrival order, averages their parameter
gradients, and steps once. The final partial window uses its actual batch count.
This changes first-stage update frequency compared with the old recomputation
loop, but prevents parameter mutation from invalidating pending graphs. The last
stage continues updating per batch. Larger windows retain more activation memory.
Positive `clip-grad-norm` clips parameter gradients immediately before each step;
zero disables clipping.

Model-specific behavior lives in `src/fine_tune/adapters.py`. To add training
support for another model, register an adapter before creating its scheduler:

```python
from src.fine_tune.adapters import TrainingAdapter, register_adapter

register_adapter("NewModel", TrainingAdapter(
    uses_attention_mask=True, token_loss=True, shift_labels=True,
))
```

The default adapter expects tensor output for classification, or `(output, mask)`
when `uses_attention_mask=True`. For other signatures, supply an adapter with
`forward(model, inputs, attention_mask)`, `loss(output, labels)`, `label_key`, and
`uses_attention_mask`. Model construction, LoRA targets, dataset and validation
support still need wiring into their existing modules; no scheduler copy is
needed. Existing loss semantics are preserved: BERT classification, GPT-2 shifted
token loss ignoring its EOS padding ID 50256, and Llama unshifted input-token loss.
The testbed still supports two model stages.

Run the offline CPU regression tests after installing the requirements:

```commandline
python -m unittest discover -s tests -v
```

These tests use in-memory RabbitMQ messages and small instances of the actual
model implementations. They check forward counts, dropout gradient equivalence,
out-of-order replies, bounded windows, partial windows, round resets, masks, loss
contracts, and last-stage gradients/updates. They do not require model downloads.
