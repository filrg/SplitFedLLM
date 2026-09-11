"""Coordinator for independent U-shaped training groups and round-wise FedAvg."""

import os
import math
import secrets
import pickle
import uuid

import torch

from src.Server import Server
from src.Utils import fed_avg_state_dicts
from src.model.u_shape import model_class, validate_cuts, split_state_dict, join_state_dict


class UShapeServer(Server):
    def __init__(self, config):
        counts = config["server"]["clients"]
        if len(counts) != 2 or any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("U-shape clients must be positive [owners, workers] counts")
        self.u_config = config["learning"].get("u-shape", {})
        self.tail_blocks = config["server"].get("tail-blocks", 0)
        name = config["server"]["model-name"]
        validate_cuts(config["server"]["cut-layers"],
                      config["server"]["model"][name]["n_block"], self.tail_blocks)
        super().__init__(config)
        self.members = {}
        self.endpoints = {}
        self.compute_weights = {}
        self.updates = {}
        self.session = None
        self.groups = {}
        self.initial = None

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        action = message["action"]
        node = str(message["client_id"])
        if action == "REGISTER":
            layer = message["layer_id"]
            if layer not in (1, 2):
                raise ValueError("U-shape layer_id must be 1 (owner) or 2 (worker)")
            if node in self.members and self.members[node] != layer:
                raise ValueError("Client attempted to change role")
            if node not in self.members:
                if sum(v == layer for v in self.members.values()) >= self.total_clients[layer - 1]:
                    raise ValueError("Too many clients for the configured role")
                weight = message.get("compute_weight", 1.0)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("compute_weight must be finite and positive")
                self.compute_weights[node] = float(weight)
                if self.u_config.get("transport", "rabbitmq") == "tcp":
                    endpoint = message.get("tensor_endpoint")
                    if (not isinstance(endpoint, dict) or not endpoint.get("host")
                            or type(endpoint.get("port")) is not int or not 0 < endpoint["port"] < 65536):
                        raise ValueError("TCP clients must register a reachable tensor endpoint")
                    self.endpoints[node] = dict(endpoint, weight=weight)
                self.members[node] = layer
            if self.session is None and len(self.members) == sum(self.total_clients):
                self._start_round()
        elif message.get("session") != self.session or node not in self.members:
            pass  # Ignore late updates from an earlier run/round.
        elif action == "FAILED":
            self._stop(f"U-shape round failed on {node}; restart from the last checkpoint")
        elif action == "UPDATE":
            if not message["result"]:
                self._stop("U-shape round failed")
            elif node not in self.updates:
                if message["layer_id"] != self.members[node] or message["size"] < 0:
                    raise ValueError("Invalid round update")
                self.updates[node] = message
                if len(self.updates) == len(self.members):
                    self._finish_round()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _start_round(self):
        owners = sorted(node for node, role in self.members.items() if role == 1)
        workers = sorted(node for node, role in self.members.items() if role == 2)
        # Every owner can dispatch microbatches to every body replica via WRR.
        self.groups = {worker: owners for worker in workers}
        largest_group = max(map(len, self.groups.values()))
        worker_limit = self.u_config.get("worker-max-inflight", max(8, largest_group))
        if worker_limit < largest_group:
            raise ValueError("worker-max-inflight must be >= the largest group's owner count")
        if self.initial is None:
            path = f"{self.model_name}.pt"
            if self.load_parameters and os.path.exists(path):
                full = torch.load(path, map_location="cpu", weights_only=True)
            else:
                full = model_class(self.model_name)(layer_id=0, n_block=self.total_block).state_dict()
            self.initial = split_state_dict(full, self.cut_layers, self.total_block, self.tail_blocks)
        self.session = uuid.uuid4().hex
        self.updates = {}
        self.tensor_key = secrets.token_hex(32)
        for node in (*owners, *workers):
            is_owner = node in owners
            worker = workers[0] if is_owner else node
            response = dict(
                action="START", message="Start U-shape round", architecture="u-shape",
                session=self.session, role="owner" if is_owner else "worker",
                parameters=self.initial[0 if is_owner else 1], worker_id=worker, owner_ids=owners,
                worker_ids=workers,
                worker_weights={w: getattr(self, "compute_weights", {}).get(w, 1) for w in workers},
                adaptive_weights=self.u_config.get("adaptive-weights", True),
                transport=self.u_config.get("transport", "rabbitmq"),
                tensor_key=self.tensor_key, tensor_endpoints=getattr(self, "endpoints", {}),
                model_name=self.model_name, data_name=self.data_name, num_sample=self.num_sample,
                cut_layers=self.cut_layers, tail_blocks=self.tail_blocks, total_block=self.total_block,
                batch_size=self.batch_size, lr=self.lr, weight_decay=self.weight_decay,
                clip_grad_norm=self.clip_grad_norm, fine_tune_config=self.fine_tune_config,
                validation=self.validation,
                max_inflight=self.u_config.get("max-inflight", self.control_count),
                microbatches_per_window=self.u_config.get("microbatches-per-window", 8),
                worker_max_inflight=worker_limit,
                timeout=self.u_config.get("timeout-seconds", 120),
            )
            self.send_to_response(node, pickle.dumps(response))

    def _finish_round(self):
        # Empty owners/workers do not contribute a zero denominator to FedAvg.
        averaged = []
        for role in (1, 2):
            updates = [self.updates[node] for node, layer in self.members.items()
                       if layer == role and self.updates[node]["size"] > 0]
            averaged.append(fed_avg_state_dicts([u["parameters"] for u in updates],
                                               [u["size"] for u in updates])
                            if updates else self.initial[role - 1])
        self.initial = tuple(averaged)
        if self.save_parameters:
            full = join_state_dict(*self.initial, self.cut_layers, self.total_block, self.tail_blocks)
            path = f"{self.model_name}.pt"
            torch.save(full, path + ".tmp")
            os.replace(path + ".tmp", path)
        self.round -= 1
        if self.round > 0:
            self._start_round()
        else:
            self._stop("All U-shape rounds completed", abort=False)

    def _stop(self, reason, abort=True):
        if abort:
            # Wake every training/evaluation participant, including other groups.
            for worker, owners in self.groups.items():
                for phase in ("train", "eval"):
                    for node in (*owners, worker):
                        owner = owners[0] if node == worker else node
                        queue = f"ushape_{self.session}_{phase}_{node}_ABORT"
                        self.reply_channel.queue_declare(queue=queue, durable=False)
                        self.reply_channel.basic_publish(exchange='', routing_key=queue, body=pickle.dumps(dict(
                            protocol=1, session=f"{self.session}_{phase}", window=0,
                            type="ABORT", owner=owner, worker=worker, microbatch=-1)))
        for node in self.members:
            self.send_to_response(node, pickle.dumps(dict(action="STOP", message=reason, parameters=None)))
        self.logger.log_info(reason)
        self.channel.stop_consuming()
