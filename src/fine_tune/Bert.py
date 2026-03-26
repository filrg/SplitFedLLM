import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import src.Log
from src.Optimizer import OptimizationBundle


class Ft_Bert:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id  = client_id
        self.layer_id   = layer_id
        self.channel    = channel
        self.device     = device
        self.data_count = 0
        self.size       = None

    # ── Giao tiếp ────────────────────────────────────────────

    def send_intermediate_output(self, data_id, q_numpy, scale, labels, trace):
        fwd_q = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(fwd_q, durable=False)
        trace_out = list(trace) + [self.client_id] if trace else [self.client_id]
        msg = pickle.dumps({
            "data_id": data_id,
            "data":    q_numpy,
            "scale":   scale,
            "label":   labels.cpu(),
            "trace":   trace_out,
        })
        if self.size is None:
            self.size = len(msg)
            print(f"Length message: {self.size} bytes.")
        self.channel.basic_publish(exchange="", routing_key=fwd_q, body=msg)

    def send_gradient(self, data_id, gradient, trace):
        to_id = trace[-1]
        trace = trace[:-1]
        bwd_q = f"gradient_queue_{self.layer_id - 1}_{to_id}"
        self.channel.queue_declare(queue=bwd_q, durable=False)
        msg = pickle.dumps({
            "data_id": data_id,
            "data":    gradient.detach().cpu().numpy(),
            "trace":   trace,
        })
        if self.size is None:
            self.size = len(msg)
        self.channel.basic_publish(exchange="", routing_key=bwd_q, body=msg)

    def send_to_server(self, message):
        self.channel.queue_declare("rpc_queue", durable=False)
        self.channel.basic_publish(
            exchange="", routing_key="rpc_queue", body=pickle.dumps(message)
        )

    # ── Layer 1 ───────────────────────────────────────────────

    def first_layer(self, model, lr, weight_decay, clip_grad_norm,
                    control_count=1, train_loader=None,
                    opt: OptimizationBundle = None):

        if opt is None:
            opt = OptimizationBundle({})

        optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        bwd_q_name = f"gradient_queue_{self.layer_id}_{self.client_id}"
        self.channel.queue_declare(queue=bwd_q_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        model = model.to(self.device)

        data_iter  = iter(train_loader)
        forward_t  = []
        backward_t = []
        comm_t     = []
        num_fwd = num_bwd = 0
        end_data   = False
        data_store = {}  # {uuid: input_ids_tensor}

        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                model.train()

                # ── Backward pass nếu có gradient ──────────
                method_frame, _, body = self.channel.basic_get(
                    queue=bwd_q_name, auto_ack=True
                )
                if method_frame and body:
                    num_bwd += 1
                    recv       = pickle.loads(body)
                    gradient   = torch.tensor(recv["data"]).to(self.device)
                    data_id    = recv["data_id"]
                    data_input = data_store.pop(data_id)  # Fix Bug 7: xóa ngay sau dùng

                    t0 = time.time()
                    optimizer.zero_grad()
                    with opt.precision_ctx(self.device):
                        output = model(input_ids=data_input)

                    # Fix Bug 1: chỉ gọi backward 1 lần với gradient từ layer sau
                    # KHÔNG gọi opt.step() ở đây vì đây là split backward
                    output.backward(gradient=gradient)
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                    # Fix Bug 1: scaler.step nếu FP16, không thì optimizer.step thường
                    if opt.scaler is not None:
                        opt.scaler.step(optimizer)
                        opt.scaler.update()
                    else:
                        optimizer.step()

                    backward_t.append(time.time() - t0)

                else:
                    # Fix Bug 7: giới hạn data_store để tránh OOM
                    if len(data_store) > control_count:
                        continue
                    try:
                        batch   = next(data_iter)
                        inp_ids = batch["input_ids"].to(self.device)
                        labels  = batch["labels"].to(self.device)
                        data_id = uuid.uuid4()
                        data_store[data_id] = inp_ids

                        t0 = time.time()
                        with opt.precision_ctx(self.device):
                            inter_out = model(input_ids=inp_ids)
                        forward_t.append(time.time() - t0)

                        inter_out = inter_out.detach().requires_grad_(True)
                        num_fwd += 1
                        self.data_count += 1
                        pbar.update(1)

                        t0 = time.time()
                        q_numpy, scale = opt.quant(inter_out)
                        self.send_intermediate_output(data_id, q_numpy, scale, labels, trace=None)
                        comm_t.append(time.time() - t0)

                    except StopIteration:
                        end_data = True

                if end_data and num_fwd == num_bwd:
                    break

        notify = {
            "action": "NOTIFY", "client_id": self.client_id,
            "layer_id": self.layer_id, "message": "Finish training!",
        }
        src.Log.print_with_color("[>>>] Finish training!", "red")
        self.send_to_server(notify)

        bcast_q = f"reply_{self.client_id}"
        while True:
            _, _, body = self.channel.basic_get(queue=bcast_q, auto_ack=True)
            if body:
                recv = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] {recv}", "blue")
                if recv["action"] == "PAUSE":
                    print(f"Forward:  {forward_t}")
                    print(f"Backward: {backward_t}")
                    print(f"Comm:     {comm_t}")
                    return True, self.data_count
            time.sleep(0.5)

    # ── Layer 2 ───────────────────────────────────────────────

    def last_layer(self, model, lr, weight_decay, clip_grad_norm,
                   opt: OptimizationBundle = None):

        if opt is None:
            opt = OptimizationBundle({})

        optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion  = nn.CrossEntropyLoss()
        result     = True

        fwd_q_name = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=fwd_q_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print("Waiting for intermediate output. To exit press CTRL+C")
        model.to(self.device)
        model.train()

        exec_t = []
        comm_t = []

        while True:
            method_frame, _, body = self.channel.basic_get(
                queue=fwd_q_name, auto_ack=True
            )
            if method_frame and body:
                optimizer.zero_grad()
                recv    = pickle.loads(body)
                trace   = recv["trace"]
                data_id = recv["data_id"]
                labels  = recv["label"].to(self.device)

                inter_out = opt.dequant(
                    recv["data"], recv.get("scale"), self.device, requires_grad=True
                )

                t0 = time.time()
                with opt.precision_ctx(self.device):
                    output = model(input_ids=inter_out)
                    loss   = criterion(output, labels)

                if torch.isnan(loss):
                    src.Log.print_with_color("NaN detected in loss", "yellow")
                    result = False

                print(f"Loss: {loss.item():.4f}")

                # Fix Bug 3: retain_grad() TRƯỚC backward
                inter_out.retain_grad()

                # Fix Bug 2: backward + clip + step chỉ 1 lần, không lồng opt.step
                if opt.scaler is not None:
                    opt.scaler.scale(loss).backward()
                    opt.scaler.unscale_(optimizer)
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    opt.scaler.step(optimizer)
                    opt.scaler.update()
                else:
                    loss.backward()
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()

                exec_t.append(time.time() - t0)
                self.data_count += 1

                # Fix Bug 3: inter_out.grad giờ đã có sau backward
                t0 = time.time()
                self.send_gradient(data_id, inter_out.grad, trace)
                comm_t.append(time.time() - t0)

            else:
                bcast_q = f"reply_{self.client_id}"
                _, _, body = self.channel.basic_get(queue=bcast_q, auto_ack=True)
                if body:
                    recv = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] {recv}", "blue")
                    if recv["action"] == "PAUSE":
                        print(f"Exec: {exec_t}")
                        print(f"Comm: {comm_t}")
                        return result, self.data_count

    def train_on_middle_layer(self, *args, **kwargs): pass
    def alone_training(self, *args, **kwargs): pass
