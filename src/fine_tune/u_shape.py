"""Event-driven U-shape scheduler. A worker coordinates its owners' update barrier.

No input IDs, labels, logits or per-example losses enter the wire protocol.
Graphs are process-local; a failed process requires restarting the round.
"""

import pickle
import time

import torch

from src.fine_tune.adapters import get_adapter


FIELDS = {
    "PLAN": {"count", "last"},
    "OPEN": {"count", "credit"},
    "BODY_FORWARD": {"tensor", "padding_mask"},
    "TAIL_FORWARD": {"tensor"},
    "BODY_BACKWARD": {"tensor"},
    "FRONT_BACKWARD": {"tensor"},
    "EVAL_DONE": set(),
    "FRONT_DONE": set(),
    "DRAINED": set(),
    "COMMIT": {"last"},
    "ABORT": set(),
}
ENVELOPE = {"protocol", "session", "window", "type", "owner", "worker", "microbatch"}
OWNER_MESSAGES = {"PLAN", "BODY_FORWARD", "BODY_BACKWARD", "EVAL_DONE", "DRAINED"}


class UShapeScheduler:
    def __init__(self, model_name, client_id, channel, device, *, session,
                 worker_id, owner_ids, max_inflight=2, microbatches_per_window=8,
                 worker_max_inflight=8, timeout=120):
        self.client_id = str(client_id)
        self.worker_id = str(worker_id)
        self.owners = tuple(map(str, owner_ids))
        if not self.owners or len(set(self.owners)) != len(self.owners):
            raise ValueError("A worker needs a nonempty, unique owner list")
        if self.worker_id in self.owners or self.client_id not in (*self.owners, self.worker_id):
            raise ValueError("Invalid U-shape membership")
        for value in (max_inflight, microbatches_per_window, worker_max_inflight):
            if type(value) is not int or value < 1:
                raise ValueError("U-shape window and credit limits must be positive integers")
        if worker_max_inflight < len(self.owners):
            raise ValueError("worker_max_inflight must reserve at least one graph per owner")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.channel = channel
        self.device = device
        self.session = str(session)
        self.window = 0
        self.adapter = get_adapter(model_name)
        self.quota = microbatches_per_window
        self.credit = min(max_inflight, worker_max_inflight // len(self.owners))
        self.worker_limit = worker_max_inflight
        self.timeout = timeout
        self.declared = set()
        self.seen = set()
        self.data_count = 0  # samples, used for FedAvg (not number of messages)
        self.stats = {"front_forward": 0, "tail_forward": 0, "body_forward": 0,
                      "front_backward": 0, "body_backward": 0, "peak_pending": 0,
                      "steps": 0, "wire_bytes": 0}
        self.last_activity = time.monotonic()
        self.local_metrics = {"loss_sum": 0.0, "batches": 0}

    def _queue(self, node, kind):
        queue = f"ushape_{self.session}_{node}_{kind}"
        if queue not in self.declared:
            self.channel.queue_declare(queue=queue, durable=False)
            self.declared.add(queue)
        return queue

    def _send(self, kind, owner, microbatch=-1, **payload):
        if set(payload) != FIELDS[kind]:
            raise ValueError(f"Invalid fields for {kind}: {set(payload)}")
        message = dict(protocol=1, session=self.session, window=self.window,
                       type=kind, owner=owner, worker=self.worker_id, microbatch=microbatch)
        for key, value in payload.items():
            message[key] = value.detach().cpu() if torch.is_tensor(value) else value
        target = self.worker_id if kind in OWNER_MESSAGES else owner
        if kind == "ABORT" and self.client_id != self.worker_id:
            target = self.worker_id
        wire = pickle.dumps(message)
        self.channel.basic_publish(exchange='', routing_key=self._queue(target, kind), body=wire)
        self.stats["wire_bytes"] += len(wire)
        self.last_activity = time.monotonic()

    def _ack(self, message):
        self.channel.basic_ack(delivery_tag=message.pop("_delivery"))
        self.seen.add((message["window"], message["type"], message["owner"], message["microbatch"]))

    def _poll(self, *kinds):
        for kind in ("ABORT", *kinds):
            method, _, body = self.channel.basic_get(queue=self._queue(self.client_id, kind), auto_ack=False)
            if not method:
                continue
            message = pickle.loads(body)
            if (set(message) != ENVELOPE | FIELDS[kind] or message["type"] != kind
                    or message["protocol"] != 1 or message["session"] != self.session
                    or message["worker"] != self.worker_id or message["owner"] not in self.owners
                    or (self.client_id != self.worker_id and message["owner"] != self.client_id)):
                raise ValueError("Invalid U-shape envelope or route")
            key = (message["window"], kind, message["owner"], message["microbatch"])
            if kind != "ABORT" and (key in self.seen or message["window"] < self.window):
                self.channel.basic_ack(delivery_tag=method.delivery_tag)
                continue  # Re-delivery never triggers a second backward or step.
            if kind != "ABORT" and message["window"] != self.window:
                raise ValueError("Unexpected model/window version")
            message["_delivery"] = method.delivery_tag
            self.last_activity = time.monotonic()
            if kind == "ABORT":
                self._ack(message)
                raise RuntimeError("Peer aborted U-shape round; restart from a common checkpoint")
            return message
        return None

    def _idle(self):
        if time.monotonic() - self.last_activity > self.timeout:
            raise TimeoutError("U-shape peer timeout; round cannot commit")
        time.sleep(0.001)

    def _wait(self, kind):
        while True:
            message = self._poll(kind)
            if message is not None:
                return message
            self._idle()

    def _abort(self):
        # Never include exception text: it could contain private input/labels.
        for owner in self.owners if self.client_id == self.worker_id else (self.client_id,):
            try:
                self._send("ABORT", owner)
            except Exception:
                pass  # The original exception remains the cause of failure.

    def _tensor(self, value, *, like=None, leaf=False):
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError("Expected a floating-point hidden state or gradient")
        if like is not None and (value.shape != like.shape or value.dtype != like.dtype):
            raise ValueError("Gradient shape/dtype mismatch")
        tensor = value.to(self.device).detach()
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite hidden state or gradient")
        return tensor.requires_grad_(leaf)

    def _mask(self, padding, hidden):
        if not self.adapter.uses_attention_mask:
            return None
        batch, length = hidden.shape[:2]
        if padding is None or padding.shape != (batch, length):
            raise ValueError("Expected a B x T padding mask")
        padding = padding.to(self.device)
        causal = torch.ones(length, length, device=self.device, dtype=hidden.dtype).tril()
        return causal[None, None] * padding[:, None, None, :] * padding[:, None, :, None]

    @staticmethod
    def _optimizer(model, lr, weight_decay):
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay) if parameters else None

    def _step(self, model, optimizer, clip):
        if optimizer is not None:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        self.stats["steps"] += 1

    def owner(self, model, loader, lr, weight_decay=0, clip_grad_norm=0, training=True):
        """Run front and tail on the data owner; labels remain in local pending state."""
        if self.client_id == self.worker_id:
            raise ValueError("Worker cannot run owner role")
        model.to(self.device).train(training)
        optimizer = self._optimizer(model, lr, weight_decay) if training else None
        iterator = iter(loader)
        remaining = len(loader)
        pending = {}
        try:
            while True:
                count = min(self.quota, remaining)
                last = remaining == count
                self._send("PLAN", self.client_id, count=count, last=last)
                opened = self._wait("OPEN")
                denominator, credit = opened["count"], opened["credit"]
                if denominator < count or not 1 <= credit <= self.credit:
                    raise ValueError("Invalid group window plan")
                self._ack(opened)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                sent = completed = 0
                while completed < count:
                    # Process dependencies first, then immediately refill a free slot.
                    message = self._poll("FRONT_BACKWARD" if training else "FRONT_DONE", "TAIL_FORWARD")
                    if message is not None:
                        index = message["microbatch"]
                        if index not in pending:
                            raise ValueError("Gradient/hidden state has no matching local graph")
                        entry = pending[index]
                        if message["type"] == "TAIL_FORWARD":
                            if entry["phase"] != "tail":
                                raise ValueError("Unexpected tail transition")
                            hidden = self._tensor(message["tensor"], leaf=training)
                            with torch.set_grad_enabled(training):
                                output, _ = self.adapter.forward(model["tail"], hidden,
                                                                 self._mask(entry["padding"], hidden))
                                loss = self.adapter.loss(output, entry["labels"])
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite local loss; window aborted")
                            # Objective: mean of per-microbatch mean losses across the group.
                            # Scale ONCE here; all three partitions receive that scale.
                            self.local_metrics["loss_sum"] += loss.item()
                            self.local_metrics["batches"] += 1
                            if training:
                                (loss / denominator).backward()
                                self._send("BODY_BACKWARD", self.client_id, index, tensor=hidden.grad)
                            else:
                                self._send("EVAL_DONE", self.client_id, index)
                            entry["phase"] = "front"
                            del entry["labels"], hidden, output, loss
                            self.stats["tail_forward"] += 1
                        else:
                            if entry["phase"] != "front":
                                raise ValueError("Front gradient arrived before tail backward")
                            if training:
                                entry["output"].backward(self._tensor(message["tensor"], like=entry["output"]))
                            self.data_count += entry["size"]
                            del pending[index]
                            completed += 1
                            self.stats["front_backward"] += int(training)
                        del entry
                        self._ack(message)
                    if sent < count and len(pending) < credit:
                        batch = next(iterator)
                        inputs = batch["input_ids"].to(self.device)
                        labels = batch[self.adapter.label_key].to(self.device)
                        padding = (batch["attention_mask"].to(self.device)
                                   if self.adapter.uses_attention_mask else None)
                        with torch.set_grad_enabled(training):
                            output, _ = self.adapter.forward(model["front"], inputs, padding)
                        pending[sent] = dict(output=output, labels=labels, padding=padding,
                                             size=inputs.shape[0], phase="tail")
                        self._send("BODY_FORWARD", self.client_id, sent, tensor=output, padding_mask=padding)
                        sent += 1
                        self.stats["front_forward"] += 1
                        self.stats["peak_pending"] = max(self.stats["peak_pending"], len(pending))
                        del batch, inputs, labels, padding, output
                    elif message is None:
                        self._idle()
                if pending:
                    raise RuntimeError("Owner attempted to commit live graphs")
                self._send("DRAINED", self.client_id)
                committed = self._wait("COMMIT")
                if count and training:
                    self._step(model, optimizer, clip_grad_norm)
                done = committed["last"]
                self._ack(committed)
                remaining -= count
                self.window += 1
                self.seen.clear()
                if done:
                    if remaining:
                        raise ValueError("Group ended before owner exhausted data")
                    return True, self.data_count
        except Exception:
            pending.clear()
            self._abort()
            raise

    def worker(self, model, lr, weight_decay=0, clip_grad_norm=0, training=True):
        if self.client_id != self.worker_id:
            raise ValueError("Owner cannot run worker role")
        model.to(self.device).train(training)
        optimizer = self._optimizer(model, lr, weight_decay) if training else None
        pending = {}
        try:
            while True:
                plans = {}
                while len(plans) < len(self.owners):
                    message = self._wait("PLAN")
                    if (type(message["count"]) is not int or not 0 <= message["count"] <= self.quota
                            or type(message["last"]) is not bool):
                        raise ValueError("Invalid owner quota")
                    plans[message["owner"]] = (message["count"], message["last"])
                    self._ack(message)
                total = sum(count for count, _ in plans.values())
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                for owner in self.owners:
                    self._send("OPEN", owner, count=total, credit=self.credit)
                forwards = {owner: set() for owner in self.owners}
                backwards = {owner: 0 for owner in self.owners}
                drained = set()
                while len(drained) != len(self.owners):
                    message = self._poll("BODY_BACKWARD" if training else "EVAL_DONE", "DRAINED", "BODY_FORWARD")
                    if message is None:
                        self._idle()
                        continue
                    owner, index = message["owner"], message["microbatch"]
                    key = owner, index
                    if message["type"] == "BODY_FORWARD":
                        if owner in drained or not 0 <= index < plans[owner][0] or index in forwards[owner]:
                            raise ValueError("Forward outside owner quota")
                        outstanding = len(forwards[owner]) - backwards[owner]
                        if outstanding >= self.credit or len(pending) >= self.worker_limit:
                            raise ValueError("Owner exceeded graph credit")
                        inputs = self._tensor(message["tensor"], leaf=training)
                        padding = message["padding_mask"]
                        with torch.set_grad_enabled(training):
                            output, _ = self.adapter.forward(model, inputs, self._mask(padding, inputs))
                        pending[key] = (inputs, output) if training else inputs.shape[0]
                        forwards[owner].add(index)
                        self._send("TAIL_FORWARD", owner, index, tensor=output)
                        self.stats["body_forward"] += 1
                        self.stats["peak_pending"] = max(self.stats["peak_pending"], len(pending))
                        del inputs, output, padding
                    elif message["type"] in ("BODY_BACKWARD", "EVAL_DONE"):
                        if key not in pending:
                            raise ValueError("Body gradient has no matching forward graph")
                        if training:
                            inputs, output = pending.pop(key)
                            output.backward(self._tensor(message["tensor"], like=output))
                            self._send("FRONT_BACKWARD", owner, index, tensor=inputs.grad)
                            self.data_count += inputs.shape[0]
                            del inputs, output
                        else:
                            self.data_count += pending.pop(key)
                            self._send("FRONT_DONE", owner, index)
                        backwards[owner] += 1
                        self.stats["body_backward"] += int(training)
                    else:
                        if backwards[owner] != plans[owner][0]:
                            raise ValueError("Owner drained before its backwards completed")
                        drained.add(owner)
                    self._ack(message)
                if pending:
                    raise RuntimeError("Worker attempted to update live graphs")
                if total and training:
                    self._step(model, optimizer, clip_grad_norm)
                done = all(last for _, last in plans.values())
                for owner in self.owners:
                    self._send("COMMIT", owner, last=done)
                self.window += 1
                self.seen.clear()
                if done:
                    return True, self.data_count
        except Exception:
            pending.clear()
            self._abort()
            raise
