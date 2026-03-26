from src.model.GPT2 import GPT2
from src.model.Llama import Llama
from src.model.Bert import Bert
import torch
import os
import random
import pika
import pickle
import sys
import numpy as np
import src.Log
import src.Utils
from src.val.get_val import get_val


class Server:
    def __init__(self, config):
        address      = config["rabbit"]["address"]
        username     = config["rabbit"]["username"]
        password     = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name      = config["server"]["model-name"]
        self.data_name       = config["server"]["data-name"]
        self.total_clients   = config["server"]["clients"]
        self.cut_layers      = config["server"]["cut-layers"]
        self.global_round    = config["server"]["global-round"]
        self.round           = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation      = config["server"]["validation"]

        self.total_block     = config["server"]["model"][self.model_name]["n_block"]
        self.batch_size      = config["learning"]["batch-size"]
        self.lr              = config["learning"]["learning-rate"]
        self.weight_decay    = config["learning"]["weight-decay"]
        self.control_count   = config["learning"]["control-count"]
        self.clip_grad_norm  = config["learning"]["clip-grad-norm"]
        self.data_distribution = config["server"]["data-distribution"]

        self.non_iid           = self.data_distribution["non-iid"]
        self.num_label         = self.data_distribution["num-label"]
        self.num_sample        = self.data_distribution["num-sample"]
        self.refresh_each_round = self.data_distribution.get("refresh-each-round", False)
        self.random_seed       = config["server"].get("random-seed", 1)

        self.fine_tune_config = config["fine-tune"]
        self.opt_config       = config.get("optimization", {})
        self.config           = config

        self.model_params = {
            "vocab_size":      config["server"].get("vocab_size", 50257),
            "n_embd":          config["server"].get("n_embd", 768),
            "n_layer":         config["server"].get("n_layer", 12),
            "n_head":          config["server"].get("n_head", 12),
            "pretrained_path": config["server"].get("pretrained_path", f"{self.model_name}.pt"),
        }

        if self.random_seed:
            random.seed(self.random_seed)

        log_path    = config["log_path"]
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=address, port=5672,
                virtual_host=f"{virtual_host}",
                credentials=credentials,
                heartbeat=0,
                blocked_connection_timeout=None,
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="rpc_queue")

        self.count_notify     = 0
        self.count_ready      = 0  
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.responses        = {}
        self.list_clients     = []
        self.round_result     = True

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue="rpc_queue", on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(
            f"Application start. Server waiting for {self.total_clients} clients."
        )
        src.Log.print_with_color(
            f"Application start. Server waiting for {self.total_clients} clients.", "green"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def distribution(self):
        num_clients = sum(self.total_clients)
        if self.non_iid:
            label_dist = np.random.dirichlet(
                [self.data_distribution["dirichlet"]["alpha"]] * self.num_label,
                num_clients
            )
            self.label_counts = (label_dist * self.num_sample).astype(int)
        else:
            self.label_counts = np.full(
                (num_clients, self.num_label),
                self.num_sample // self.num_label
            )

    # ── Main handler ──────────────────────────────────────────────────────────

    def on_request(self, ch, method, props, body):
        message   = pickle.loads(body)
        action    = message["action"]
        client_id = message["client_id"]
        layer_id  = message["layer_id"]

        # ── REGISTER ──────────────────────────────────────────────────────────
        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(
                f"[<<<] REGISTER from client {client_id} layer {layer_id}", "blue"
            )
            self.register_clients[layer_id - 1] += 1

            if all(
                self.register_clients[i] >= self.total_clients[i]
                for i in range(len(self.total_clients))
            ):
                src.Log.print_with_color("All clients connected. Starting round 1.", "green")
                self.distribution()
                self.logger.log_info("Start training round 1")
                self.notify_clients()


        # Client báo hoàn thành train. Server gửi PAUSE để client biết có thể lưu LoRA.
        elif action == "NOTIFY":
            src.Log.print_with_color(
                f"[<<<] NOTIFY from client {client_id} layer {layer_id}", "blue"
            )
            self.count_notify += 1

            if self.count_notify == sum(self.total_clients):
                self.count_notify = 0
                current_round     = self.global_round - self.round + 1
                src.Log.print_with_color(
                    f"All clients finished training round {current_round}.", "yellow"
                )

                # Gửi PAUSE → client nhận xong mới lưu LoRA rồi gửi READY
                pause_msg = {
                    "action":        "PAUSE",
                    "message":       f"Round {current_round} done. Save your LoRA.",
                    "current_round": current_round,
                    "parameters":    None,
                }
                for (cid, lid) in self.list_clients:
                    self.send_to_response(cid, pickle.dumps(pause_msg))

                self.logger.log_info(f"Round {current_round} complete. Waiting for READY.")

        # Server chỉ gửi START round tiếp khi đủ tất cả READY.
        elif action == "READY":
            src.Log.print_with_color(
                f"[<<<] READY from client {client_id} layer {layer_id}", "blue"
            )
            self.count_ready += 1

            if self.count_ready == sum(self.total_clients):
                self.count_ready  = 0
                self.round       -= 1
                current_round     = self.global_round - self.round

                src.Log.print_with_color(
                    f"All clients ready. Round {current_round} fully complete.", "green"
                )
                self.logger.log_info(f"Round {current_round} fully complete.")

                # Merge GPT2_layer1.pt + GPT2_layer2.pt → GPT2.pt
                self._merge_layer_files()

                if self.round > 0:
                    next_round = self.global_round - self.round + 1
                    self.logger.log_info(f"Start training round {next_round}")
                    self.notify_clients()
                else:
                    self.logger.log_info("Stop training !!!")
                    self.notify_clients(start=False)
                    sys.exit()

        elif action == "UPDATE":
            src.Log.print_with_color(
                f"[WARN] Deprecated UPDATE from client {client_id} — ignored.", "yellow"
            )

        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            src.Log.print_with_color(f"[WARN] basic_ack failed: {e}", "yellow")
            self._reconnect()


    def notify_clients(self, start=True):
        for (client_id, layer_id) in self.list_clients:
            if not start:
                self.send_to_response(
                    client_id,
                    pickle.dumps({"action": "STOP", "message": "Stop training!", "parameters": None})
                )
                src.Log.print_with_color(f"[>>>] STOP → client {client_id}", "red")
                continue

            response = {
                "action":            "START",
                "message":           "Server accept the connection!",
                "parameters":        None,   
                "cut_layers":        self.cut_layers,
                "total_block":       self.total_block,
                "model_name":        self.model_name,
                "data_name":         self.data_name,
                "num_sample":        self.num_sample,
                "control_count":     self.control_count,
                "batch_size":        self.batch_size,
                "lr":                self.lr,
                "weight_decay":      self.weight_decay,
                "clip_grad_norm":    self.clip_grad_norm,
                "fine_tune_config":  self.fine_tune_config,
                "opt_config":        self.opt_config,
                "refresh_each_round": self.refresh_each_round,  
            }
            src.Log.print_with_color(
                f"[>>>] START → client {client_id} layer {layer_id}", "red"
            )
            self.send_to_response(client_id, pickle.dumps(response))


    def _merge_layer_files(self):
        """
        Merge GPT2_layer1.pt (wte, wpe, h.0-3) và GPT2_layer2.pt (h.4-11, ln_f, lm_head)
        thành GPT2.pt đầy đủ để round tiếp theo load.
        Layer 2 keys cần được remap: h.0 → h.{cut_layers}, h.1 → h.{cut_layers+1}, ...
        """
        f1 = f"{self.model_name}_layer1.pt"
        f2 = f"{self.model_name}_layer2.pt"

        if not os.path.exists(f1) or not os.path.exists(f2):
            src.Log.print_with_color(
                f"[WARN] Merge skipped: {f1} exists={os.path.exists(f1)}, "
                f"{f2} exists={os.path.exists(f2)}", "yellow"
            )
            return

        sd1 = torch.load(f1, map_location="cpu")
        sd2 = torch.load(f2, map_location="cpu")

        merged = {}

        for k, v in sd1.items():
            merged[k] = v

        for k, v in sd2.items():
            if k.startswith("h."):
                parts   = k.split(".")
                old_idx = int(parts[1])
                new_idx = old_idx + self.cut_layers
                new_k   = ".".join([parts[0], str(new_idx)] + parts[2:])
                merged[new_k] = v
            else:

                merged[k] = v


        if "lm_head.weight" not in merged and "wte.weight" in merged:
            merged["lm_head.weight"] = merged["wte.weight"].clone()

        out_file = f"{self.model_name}.pt"
        torch.save(merged, out_file)
        src.Log.print_with_color(
            f"[>>>] Merged {f1} + {f2} → {out_file} "
            f"({len(merged)} keys)", "green"
        )
        self.logger.log_info(f"Merged layer files → {out_file}")

    def _reconnect(self):
        src.Log.print_with_color("[>>>] Reconnecting to RabbitMQ...", "yellow")
        try:
            self.connection.close()
        except Exception:
            pass
        credentials = pika.PlainCredentials(
            self.config["rabbit"]["username"],
            self.config["rabbit"]["password"]
        )
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.config["rabbit"]["address"],
                port=5672,
                virtual_host=self.config["rabbit"]["virtual-host"],
                credentials=credentials,
                heartbeat=0,
                blocked_connection_timeout=None,
            )
        )
        self.channel       = self.connection.channel()
        self.reply_channel = self.connection.channel()
        self.channel.queue_declare(queue="rpc_queue")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue="rpc_queue", on_message_callback=self.on_request)
        src.Log.print_with_color("[>>>] Reconnected successfully.", "green")

    def start(self):
        while True:
            try:
                self.channel.start_consuming()
            except (
                pika.exceptions.ChannelWrongStateError,
                pika.exceptions.StreamLostError,
                pika.exceptions.ConnectionClosedByBroker,
                Exception,
            ) as e:
                src.Log.print_with_color(
                    f"[WARN] Connection error: {e} — reconnecting...", "yellow"
                )
                try:
                    self._reconnect()
                except Exception as re:
                    src.Log.print_with_color(f"[ERROR] Reconnect failed: {re}", "red")
                    import time; time.sleep(5)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        self.reply_channel.basic_publish(
            exchange="", routing_key=reply_queue_name, body=message
        )