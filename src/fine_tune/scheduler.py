import pickle
import time
import uuid

import torch
from tqdm import tqdm

from src.fine_tune.adapters import get_adapter


class SplitScheduler:
    def __init__(self, client_id, layer_id, channel, device, adapter):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.adapter = adapter
        self.data_count = 0
        self.timings = {}

    def _publish(self, queue, message):
        self.channel.queue_declare(queue=queue, durable=False)
        self.channel.basic_publish(exchange='', routing_key=queue,
                                   body=pickle.dumps(message))

    def send_intermediate_output(self, data_id, output, attention_mask, labels, trace=None):
        message = {"data_id": data_id, "data": output.detach().cpu().numpy(),
                   "label": labels.detach().cpu(),
                   "trace": list(trace or []) + [self.client_id]}
        if attention_mask is not None:
            message["attention_mask"] = attention_mask.detach().cpu()
        self._publish(f'intermediate_queue_{self.layer_id}', message)

    def send_gradient(self, data_id, gradient, trace):
        self._publish(f'gradient_queue_{self.layer_id - 1}_{trace[-1]}',
                      {"data_id": data_id, "data": gradient.detach().cpu().numpy(),
                       "trace": list(trace[:-1])})

    def send_to_server(self, message):
        self._publish('rpc_queue', message)

    def _receive(self, queue):
        method, _, body = self.channel.basic_get(queue=queue, auto_ack=True)
        return pickle.loads(body) if method and body else None

    def _paused(self):
        message = self._receive(f'reply_{self.client_id}')
        return message is not None and message['action'] == 'PAUSE'

    def _start(self, model, lr, weight_decay):
        self.data_count = 0
        self.timings = {"forward": [], "backward": [], "comm": []}
        model.to(self.device)
        model.train()
        self.channel.queue_declare(queue=f'reply_{self.client_id}', durable=False)
        self.channel.basic_qos(prefetch_count=1)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _step(model, optimizer, clip_grad_norm, batch_count=1):
        # Average batch gradients in a window before clipping/updating. Never
        # mutate parameters while another graph using them awaits backward.
        if batch_count > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(batch_count)
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

    def first_layer(self, model, lr, weight_decay, clip_grad_norm,
                    control_count=1, train_loader=None):
        if isinstance(control_count, bool) or not isinstance(control_count, int) or control_count < 1:
            raise ValueError('control_count must be a positive integer')
        if train_loader is None:
            raise ValueError('first_layer requires a train_loader')
        optimizer = self._start(model, lr, weight_decay)
        queue = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=queue, durable=False)
        data_iter = iter(train_loader)
        exhausted = False

        with tqdm(total=len(train_loader), desc='Processing', unit='step') as pbar:
            while not exhausted:
                pending = {}
                optimizer.zero_grad()
                for _ in range(control_count):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch[self.adapter.label_key].to(self.device)
                    mask = (batch['attention_mask'].to(self.device)
                            if self.adapter.uses_attention_mask else None)
                    started = time.perf_counter()
                    output, mask = self.adapter.forward(model, inputs, mask)
                    self.timings['forward'].append(time.perf_counter() - started)
                    data_id = uuid.uuid4()
                    # Keep the ORIGINAL output and its autograd graph locally.
                    # Only the wire representation is detached.
                    pending[data_id] = output
                    started = time.perf_counter()
                    self.send_intermediate_output(data_id, output, mask, labels)
                    self.timings['comm'].append(time.perf_counter() - started)
                    del output

                batch_count = len(pending)
                while pending:
                    message = self._receive(queue)
                    if message is None:
                        time.sleep(0.001)
                        continue
                    data_id = message['data_id']
                    if data_id not in pending:
                        raise ValueError(f'Unknown or duplicate gradient data_id: {data_id}')
                    output = pending.pop(data_id)
                    gradient = torch.as_tensor(message['data'], device=output.device,
                                               dtype=output.dtype)
                    started = time.perf_counter()
                    output.backward(gradient=gradient)
                    self.timings['backward'].append(time.perf_counter() - started)
                    del output, gradient
                    self.data_count += 1
                    pbar.update(1)
                if batch_count:
                    self._step(model, optimizer, clip_grad_norm, batch_count)

        self.send_to_server({"action": "NOTIFY", "client_id": self.client_id,
                             "layer_id": self.layer_id, "message": "Finish training!"})
        while not self._paused():
            time.sleep(0.5)
        print(f'Training timings: {self.timings}')
        return True, self.data_count

    def last_layer(self, model, lr, weight_decay, clip_grad_norm):
        optimizer = self._start(model, lr, weight_decay)
        queue = f'intermediate_queue_{self.layer_id - 1}'
        self.channel.queue_declare(queue=queue, durable=False)
        result = True
        while True:
            message = self._receive(queue)
            if message is None:
                if self._paused():
                    print(f'Training timings: {self.timings}')
                    return result, self.data_count
                time.sleep(0.001)
                continue
            optimizer.zero_grad()
            # Create a leaf directly on the target device (including CUDA).
            inputs = torch.as_tensor(message['data'], device=self.device).detach().requires_grad_(True)
            labels = message['label'].to(self.device)
            mask = message.get('attention_mask')
            if mask is not None:
                mask = mask.to(self.device)
            started = time.perf_counter()
            output, _ = self.adapter.forward(model, inputs, mask)
            loss = self.adapter.loss(output, labels)
            self.timings['forward'].append(time.perf_counter() - started)
            if not torch.isfinite(loss).all():
                result = False
            print(f'Loss: {loss.item()}')
            started = time.perf_counter()
            loss.backward()
            self._step(model, optimizer, clip_grad_norm)
            self.timings['backward'].append(time.perf_counter() - started)
            self.data_count += 1
            started = time.perf_counter()
            self.send_gradient(message['data_id'], inputs.grad, message['trace'])
            self.timings['comm'].append(time.perf_counter() - started)
            del inputs, output, loss


def create_scheduler(model_name, client_id, layer_id, channel, device):
    return SplitScheduler(client_id, layer_id, channel, device, get_adapter(model_name))
