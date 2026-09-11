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
    transport: tcp
    pinned-memory: auto
    pool-bytes: 67108864
    transport-queue-size: 32
    adaptive-weights: true
    max-inflight: 2
    microbatches-per-window: 8
    worker-max-inflight: 8
    timeout-seconds: 120
```

Run `python server.py`, then start four owner processes with
`python client.py --layer_id 1` and two worker processes with
`python client.py --layer_id 2`. Each process uses the configured broker;
`--device cpu` or `--device cuda` selects its compute device. The coordinator
registers each peer's TCP endpoint. Every owner can use every worker, including
one owner with multiple workers. Each microbatch stays pinned to the worker that
created its body graph until backward finishes.

`max-inflight` bounds live microbatches per owner. `worker-max-inflight` bounds
live body graphs across the group and must be at least its number of owners.
The effective per-owner credit is the smaller of `max-inflight` and
`floor(worker-max-inflight / owners_in_group)`. `microbatches-per-window` is a
per-owner update quota, independent of the credit limit. A released slot is
refilled immediately within the same window; backwards and tail work are serviced
before new forwards. Different owners/workers can overlap computation without
concurrent writes to a GPU's parameter gradients.

All owners and workers keep parameters unchanged until the window drains.
Each worker steps its body replica and sends COMMIT; owners wait for all worker
commits and step front+tail together before entering the next window. The objective is the **mean of per-microbatch mean
losses across the group**, scaled once at the local loss. This is not a global
token-weighted mean when batch sizes/token counts differ. Positive gradient
clipping applies separately to the owner partition and worker partition. Empty
owners and partial final windows participate in the barrier without extra steps.

Body replicas are updated independently during the round: this is not synchronous
DDP of a single body. At round end, `src/UShapeServer.py` FedAvgs
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
optimizations. The tensor data plane is now direct TCP; cross-owner tensor
batching and RDMA remain optional future work.

Run the offline regression/integration tests:

```commandline
python -m unittest discover -s tests -v
```

Tests exercise actual small BERT/GPT-2/Llama partitions, monolithic gradient and
update equivalence, dropout, graph release, LoRA merge, multiple owners, uneven
windows, reordering/duplicates, local validation, and two-round RpcClient/FedAvg
integration using an in-memory broker. No model download is needed for tests.
Real-broker/GPU throughput must be measured on the target testbed.

## Direct TCP and compute-weighted scheduling

The four activation/gradient messages use persistent direct TCP connections.
RabbitMQ carries PLAN/OPEN/DRAINED/COMMIT/ABORT and federated updates only.
Frames contain a bounded JSON header and contiguous tensor bytes, including BF16
without conversion through a NumPy numeric dtype. Tensor payloads are not pickled.
Received bytes go into reusable CPU buffers via `recv_into`.

On CUDA, D2H uses a copy stream and producer events; only the socket sender thread
waits for the copy before accessing the bytes. Receiver threads enqueue H2D and
the compute stream waits on its completion event. GPU receive leases survive
until the autograd graph completes backward. Pool reuse waits for the relevant
compute/copy events, including when a received tensor is forwarded again.

Activation and gradient traffic use separate TCP lanes, receive queues and pool
reserves, preventing a full activation queue from blocking a gradient needed to
release it. All buffers are allocated lazily and reused by shape/dtype; idle
buffers may be evicted when shapes change. `pool-bytes` limits **each** pool:
three CPU pools (send, receive activation, receive gradient), plus two CUDA receive
pools on GPU devices. Thus the default permits up to 192 MiB host + 128 MiB device
buffers, excluding model/autograd storage and socket buffers. Set budgets together
with graph credits and maximum tensor sizes; oversize frames fail explicitly and
pool exhaustion times out instead of allocating unbounded memory.

`pinned-memory: auto` probes pinned-allocation availability and falls back to
pageable staging if unavailable. It does not benchmark Jetson caching behavior.
Use `on` or `off` per device after measuring its JetPack/SoC combination. Pageable
fallback is correct but may block copies; no claim of eliminating CUDA/CPU copies
or providing RDMA is made. Dtype is preserved: this implementation does not enable
lossy FP16/INT8 conversion automatically.

Example Ethernet endpoints for one owner and two unequal workers, with
`server.clients: [1, 2]`:

```commandline
# Owner host
python client.py --layer_id 1 --device cuda --tensor-host 192.168.1.10 --tensor-port 20000
# Faster worker host
python client.py --layer_id 2 --device cuda --tensor-host 192.168.1.20 --tensor-port 20000 --compute-weight 3
# Slower worker host
python client.py --layer_id 2 --device cuda --tensor-host 192.168.1.30 --tensor-port 20000 --compute-weight 1
```

These are example addresses. Peers must be able to reach each advertised TCP
endpoint; open the chosen ports on the testbed network. `--tensor-port 0` chooses
an available port, and an omitted `--tensor-host` uses the route to the broker.
Use explicit addresses on multihomed hosts. Binding defaults to `0.0.0.0` and can
be changed with `--tensor-bind`. Each round has an authenticated session handshake
using a key distributed over the existing broker. TCP payloads are not encrypted;
this backend assumes the same trusted testbed network as the broker.

Smooth weighted round robin assigns individual microbatches before each window.
Weights 3:1 assign 75%/25% over sufficiently many equal-cost batches, rather than
only assigning more owner processes to the fast worker. Credits still bound live
graphs; an owner can dispatch an eligible batch for another worker while one
worker's credits are occupied. The owner retains raw inputs/labels locally.

With `adaptive-weights: true`, workers report processed padded tokens divided by
body forward+backward compute time. CUDA measurements use events collected at
the drained window boundary; CPU measurements use `perf_counter`. Each owner
updates an EWMA (20% new measurement) for the next window, with a 20:1 maximum
measured ratio so a slow worker retains opportunities. Initial `--compute-weight`
values apply until every worker has a measurement. `false` keeps the configured
ratio fixed for controlled experiments. WRR state persists across windows;
manual initial weights reset at a new round. Inbound ready queues also use the
registered peer weights for service fairness.

Weights describe compute capacity, not network throughput; network backpressure,
variable sequence lengths and the common window barrier can limit the achieved
wall-clock speedup. Length bucketing makes microbatch counts better approximate
work. This does not move private datasets between owners, and compute weights do
not replace the actual processed-sample weights used for FedAvg.

Per-process logs expose TCP bytes/frames, D2H/H2D CUDA timings, pool allocation and
reuse counts, and graph peaks. `transport: rabbitmq` keeps the previous tensor
transport for comparison while using the same WRR scheduler. Validate performance
on each Jetson/server; the automated suite uses real loopback TCP and skips its
CUDA test when CUDA hardware is unavailable.

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
