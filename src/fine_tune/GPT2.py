import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import src.Log
from src.Optimizer import OptimizationBundle
from transformers import GPT2Tokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR


class Ft_GPT2:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id  = client_id
        self.layer_id   = layer_id
        self.channel    = channel
        self.device     = device
        self.data_count = 0


    def send_intermediate_output(self, data_id, q_numpy, scale, mask_out, labels, trace):
        fwd_q     = f"intermediate_queue_{self.layer_id}"
        trace_out = list(trace) + [self.client_id] if trace else [self.client_id]
        self.channel.queue_declare(fwd_q, durable=False)
        msg = pickle.dumps({
            "data_id": data_id,
            "data":    q_numpy,
            "scale":   scale,
            "label":   labels.cpu(),
            "trace":   trace_out,
            "mask":    mask_out.cpu(),
        })
        print(f"len message: {len(msg)} bytes")
        self.channel.basic_publish(exchange="", routing_key=fwd_q, body=msg)

    def send_end_signal(self):
        fwd_q = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(fwd_q, durable=False)
        self.channel.basic_publish(
            exchange="", routing_key=fwd_q,
            body=pickle.dumps({"action": "END"})
        )
        src.Log.print_with_color("[>>>] Client 1 gửi END signal cho client 2", "yellow")

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
        self.channel.basic_publish(exchange="", routing_key=bwd_q, body=msg)

    def send_to_server(self, message):
        self.channel.queue_declare("rpc_queue", durable=False)
        self.channel.basic_publish(
            exchange="", routing_key="rpc_queue", body=pickle.dumps(message)
        )



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

        forward_t  = []
        backward_t = []
        comm_t     = []
        data_iter  = iter(train_loader)
        num_fwd = num_bwd = 0
        end_data   = False
        data_store = {}
        scheduler  = CosineAnnealingLR(
            optimizer, T_max=max(len(train_loader), 1), eta_min=lr / 10
        )

        with tqdm(total=len(train_loader), desc="Layer 1", unit="step") as pbar:
            while True:
                model.train()

                method_frame, _, body = self.channel.basic_get(
                    queue=bwd_q_name, auto_ack=True
                )
                if method_frame and body:
                    num_bwd  += 1
                    recv      = pickle.loads(body)
                    gradient  = torch.tensor(recv["data"]).to(self.device)
                    # FIX: cast gradient về đúng dtype của inter (fp16/fp32)
                    gradient  = gradient.to(dtype=inter.dtype)
                    data_id   = recv["data_id"]
                    inter     = data_store.pop(data_id)

                    if torch.isnan(gradient).any():
                        print("[ERROR] Skip backward layer 1 (NaN gradient)")
                        continue

                    optimizer.zero_grad()
                    inter.backward(gradient)

                    has_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in model.parameters()
                    )
                    if has_nan:
                        print("[ERROR] NaN grad → skip step")
                        optimizer.zero_grad()
                        continue

                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    pbar.update(1)

                else:

                    if len(data_store) >= control_count:
                        continue
                    try:
                        batch   = next(data_iter)
                        inp_ids = batch["input_ids"].to(self.device)
                        attn    = batch["attention_mask"].to(self.device)
                        labels  = batch["labels"].to(self.device)
                        data_id = uuid.uuid4()

                        t0 = time.time()
                        with opt.precision_ctx(self.device):
                            out      = model(input_ids=inp_ids, attention_mask=attn)
                            inter    = out["hidden_states"]
                            mask_out = out["mask"]
                            inter    = inter.detach().requires_grad_(True)
                            data_store[data_id] = inter
                        forward_t.append(time.time() - t0)

                        num_fwd += 1
                        self.data_count += 1

                        t0 = time.time()
                        q_numpy, scale = opt.quant(inter)
                        self.send_intermediate_output(
                            data_id, q_numpy, scale, mask_out, labels, trace=None
                        )
                        comm_t.append(time.time() - t0)

                    except StopIteration:
                        end_data = True

                if end_data and num_fwd == num_bwd:
                    break

        self.send_end_signal()

        notify = {
            "action":    "NOTIFY",
            "client_id": self.client_id,
            "layer_id":  self.layer_id,
            "message":   "Finish training!",
        }
        src.Log.print_with_color("[>>>] Client 1 gửi NOTIFY về server", "red")
        self.send_to_server(notify)

        # Chờ PAUSE từ server
        bcast_q = f"reply_{self.client_id}"
        while True:
            _, _, body = self.channel.basic_get(queue=bcast_q, auto_ack=True)
            if body:
                recv = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Client 1: {recv['action']}", "blue")
                if recv["action"] == "PAUSE":
                    src.Log.print_with_color(
                        f"[INFO] Forward: {len(forward_t)} steps, "
                        f"Backward: {num_bwd} steps", "green"
                    )
                    return True, self.data_count
            time.sleep(0.5)



    def last_layer(self, model, lr, weight_decay, clip_grad_norm,
                   opt: OptimizationBundle = None):

        if opt is None:
            opt = OptimizationBundle({})

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        result    = True

        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=lr / 10)

        fwd_q_name = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=fwd_q_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        src.Log.print_with_color("Layer 2: Waiting for hidden states...", "green")
        model.to(self.device)
        model.train()

        exec_t        = []
        comm_t        = []
        num_received  = 0
        num_grad_sent = 0
        end_received  = False
        nan_count     = 0

        while True:
            method_frame, _, body = self.channel.basic_get(
                queue=fwd_q_name, auto_ack=True
            )
            if method_frame and body:
                recv = pickle.loads(body)

                # END sentinel
                if recv.get("action") == "END":
                    src.Log.print_with_color(
                        f"[<<<] END received. Received {num_received} batches, "
                        f"sent {num_grad_sent} gradients.", "yellow"
                    )
                    end_received = True
                    continue

                mask    = recv["mask"].to(self.device)
                trace   = recv["trace"]
                data_id = recv["data_id"]
                labels  = recv["label"].to(self.device)
                num_received += 1

                inter = opt.dequant(
                    recv["data"], recv.get("scale"), self.device, requires_grad=True
                )
                # FIX: đồng bộ dtype của inter với model để tránh Half/Float mismatch
                model_dtype = next(model.parameters()).dtype
                inter = inter.to(dtype=model_dtype)

                optimizer.zero_grad()

                t0 = time.time()
                with opt.precision_ctx(self.device):
                    out          = model(input_ids=inter, attention_mask=mask)
                    output       = out["logits"]
                    shift_logits = output[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss         = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                if torch.isnan(loss):
                    src.Log.print_with_color("NaN loss — sending zero gradient", "yellow")
                    nan_count += 1
                    num_grad_sent += 1
                    self.send_gradient(data_id, torch.zeros_like(inter), trace)
                    continue

                print(f"Loss: {loss.item():.4f}")

                inter.retain_grad()

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

                scheduler.step()
                exec_t.append(time.time() - t0)
                self.data_count += 1


                t0   = time.time()
                grad = inter.grad if inter.grad is not None else torch.zeros_like(inter)
                if inter.grad is None:
                    src.Log.print_with_color(
                        "[WARN] inter.grad is None — sending zero gradient", "yellow"
                    )
                self.send_gradient(data_id, grad, trace)
                num_grad_sent += 1
                comm_t.append(time.time() - t0)

            else:

                if end_received and num_grad_sent == num_received:
                    if num_received > 0 and nan_count / num_received > 0.5:
                        result = False
                        src.Log.print_with_color(
                            f"[WARN] {nan_count}/{num_received} batches NaN → round failed",
                            "yellow"
                        )

                    src.Log.print_with_color(
                        f"[>>>] Tất cả {num_grad_sent} gradient đã gửi. Gửi NOTIFY.", "red"
                    )
                    notify = {
                        "action":    "NOTIFY",
                        "client_id": self.client_id,
                        "layer_id":  self.layer_id,
                        "message":   "Finish training!",
                    }
                    self.send_to_server(notify)
                    end_received = False

                # Chờ PAUSE từ server
                bcast_q = f"reply_{self.client_id}"
                _, _, body = self.channel.basic_get(queue=bcast_q, auto_ack=True)
                if body:
                    recv = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Client 2: {recv['action']}", "blue")
                    if recv["action"] == "PAUSE":
                        src.Log.print_with_color(
                            f"[INFO] Exec: {len(exec_t)} steps", "green"
                        )
                        return result, self.data_count

                time.sleep(0.1)

    def train_on_middle_layer(self, *args, **kwargs): pass
    def alone_training(self, *args, **kwargs): passs