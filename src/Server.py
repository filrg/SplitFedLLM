import torch
import os
import time
import random
import pika
import pickle
import sys
import numpy as np
import copy
import src.Log
import src.Utils

from src.model.Bert import Bert
from src.val.get_val import get_val

class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model-name"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.cut_layers = config["server"]["cut-layers"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]

        # Clients
        self.total_block = config["server"]["model"][self.model_name]["n_block"]
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.weight_decay = config["learning"]["weight-decay"]
        self.control_count = config["learning"]["control-count"]
        self.clip_grad_norm = config["learning"]["clip-grad-norm"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
        self.refresh_each_round = self.data_distribution["refresh-each-round"]
        self.random_seed = config["server"]["random-seed"]
        self.label_counts = None

        # Fine tune config
        self.fine_tune_config = config['fine-tune']

        # Bottleneck + Quantization
        self.bottleneck_config = config['bottleneck']

        if self.random_seed:
            random.seed(self.random_seed)

        log_path = config["log_path"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.count_update = [0 for _ in range(len(self.total_clients))]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.count_notify = 0
        self.responses = {}
        self.list_clients = []
        self.round_result = True

        self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
        self.global_client_sizes = [[] for _ in range(len(self.total_clients))]
        self.avg_state_dict = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def distribution(self):
        if self.non_iid:
            # label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.num_label,
            #                                          self.total_clients[0])
            label_distribution = np.array([[0.05, 0.05, 0.05, 0.85],
                                           [0.05, 0.05, 0.85, 0.05],
                                           [0.05, 0.85, 0.05, 0.05],
                                           [0.85, 0.05, 0.05, 0.05]
                                           ])
            self.label_counts = (label_distribution * self.num_sample).astype(int)

        else:
            self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")

                self.distribution()

                self.logger.log_info(f"Start training round 1")
                self.notify_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}

            self.count_notify += 1

            if self.count_notify == self.total_clients[0]:
                self.count_notify = 0
                src.Log.print_with_color(f"Received all the finish training notification", "yellow")

                for (client_id, layer_id) in self.list_clients:
                    self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            # self.distribution()
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")

            self.count_update[layer_id - 1] += 1
            if not result:
                self.round_result = False

            # Save client's model parameters
            if self.save_parameters and self.round_result:
                model_state_dict = message["parameters"]
                client_size = message["size"]
                self.global_model_parameters[layer_id - 1].append(model_state_dict)
                self.global_client_sizes[layer_id - 1].append(client_size)

            # If consumed all client's parameters
            if self.count_update == self.total_clients:
                src.Log.print_with_color("Collected all parameters.", "yellow")
                if self.save_parameters and self.round_result:

                    self.avg_all_parameters()
                    self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
                    self.global_client_sizes = [[] for _ in range(len(self.total_clients))]

                self.count_update = [0 for _ in range(len(self.total_clients))]
                # Test
                if self.save_parameters and self.validation and self.round_result:
                    state_dict_full = self.concatenate()

                    if not get_val(self.avg_state_dict, self.cut_layers, self.bottleneck_config, self.logger):
                        self.logger.log_warning("Training failed!")
                    else:
                        # Save to files
                        torch.save(state_dict_full, f'{self.model_name}.pt')
                        self.round -= 1
                    self.avg_state_dict = []
                else:
                    self.round -= 1

                # Start a new training round
                self.round_result = True

                if self.round > 0:
                    self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                    if self.save_parameters:
                        self.notify_clients()
                    else:
                        self.notify_clients(register=False)
                else:
                    self.logger.log_info("Stop training !!!")
                    self.notify_clients(start=False)
                    sys.exit()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True):

        # Send message to clients when consumed all clients
        klass = Bert
        stt = -1

        for (client_id, layer_id) in self.list_clients:
            # Read parameters file
            filepath = f'{self.model_name}.pt'
            bottleneck_path = f'bottleneck.pt'
            state_dict = None
            bottleneck_state_dict = None

            if start:
                if self.load_parameters and register:
                    if os.path.exists(filepath):
                        full_state_dict = torch.load(filepath, weights_only=True)

                        if layer_id == 1:
                            if self.bottleneck_config["enable"]:
                                model = klass(layer_id=1, n_block=self.cut_layers, reduce_comm=True,
                                              bottleneck_dim=self.bottleneck_config['bottleneck_dim'])
                            else:
                                model = klass(layer_id=1, n_block=self.cut_layers)
                            state_dict = model.state_dict()
                            keys = state_dict.keys()

                            for key in keys:
                                if key in full_state_dict:
                                    state_dict[key] = full_state_dict[key]

                        else:
                            if self.bottleneck_config["enable"]:
                                model = klass(layer_id=2, n_block=12 - self.cut_layers, reduce_comm=True,
                                              bottleneck_dim=self.bottleneck_config['bottleneck_dim'])
                            else:
                                model = klass(layer_id=2, n_block=12 - self.cut_layers)
                            state_dict = model.state_dict()
                            state_dict = src.Utils.change_keys(state_dict, self.cut_layers, True)
                            keys = state_dict.keys()

                            for key in keys:
                                if key in full_state_dict:
                                    state_dict[key] = full_state_dict[key]

                            state_dict =src.Utils.change_keys(state_dict, self.cut_layers, False)
                            src.Log.print_with_color(f"Load pretrain model successfully", "green")

                    else:
                        src.Log.print_with_color(f"File {filepath} does not exist.", "yellow")
                        self.logger.log_info(f"File {filepath} does not exist.")

                    if os.path.exists(bottleneck_path):
                        bottleneck_state_dict = torch.load(bottleneck_path, weights_only=True, map_location=torch.device("cpu"))
                        src.Log.print_with_color(f"Load pretrain bottleneck model", "green")
                    else:
                        src.Log.print_with_color(f"File {bottleneck_path} does not exist.", "yellow")

                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "parameters": copy.deepcopy(state_dict),
                            "bottleneck": copy.deepcopy(bottleneck_state_dict),
                            "cut_layers": self.cut_layers,
                            "total_block": self.total_block,
                            "fine_tune_config": self.fine_tune_config,
                            "bottleneck_config": self.bottleneck_config,
                            }

                self.send_to_response(client_id, pickle.dumps(response))


            else:
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!"
                            }
                self.send_to_response(client_id, pickle.dumps(response))

        time.sleep(5)
        if start:
            for (client_id, layer_id) in self.list_clients:
                if layer_id == 1:
                    stt += 1
                response = {"action": "SYN",
                            "label_counts": self.label_counts,
                            "control_count": self.control_count,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "weight_decay": self.weight_decay,
                            "stt": stt,
                            "message": "Synchronize client devices",
                            }
                self.send_to_response(client_id, pickle.dumps(response))

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def avg_all_parameters(self):
        layer_sizes = self.global_client_sizes
        layer_params = self.global_model_parameters

        for layer_idx, list_state_dicts in enumerate(layer_params):
            list_sizes = layer_sizes[layer_idx]
            if not list_state_dicts or not list_sizes:
                self.avg_state_dict.append({})
                continue
            avg_sd = src.Utils.fed_avg_state_dicts(list_state_dicts, weights=list_sizes)
            self.avg_state_dict.append(avg_sd)

    def concatenate(self):
        avg_layers = self.avg_state_dict
        if not avg_layers:
            print(f"Warning: don't has averaged layers, skipping.")

        full_dict = {}
        for idx, layer_dict in enumerate(avg_layers):
            if idx == 0:
                sd = layer_dict
                full_dict.update(copy.deepcopy(sd))
            else:
                sd = src.Utils.change_keys(layer_dict, self.cut_layers, True)
                full_dict.update(copy.deepcopy(sd))

        return full_dict
