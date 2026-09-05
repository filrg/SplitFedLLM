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

## Shared training scheduler

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
