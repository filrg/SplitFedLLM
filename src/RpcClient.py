import time
import pickle
import copy

import src.Log
from src.fine_tune.BERT import Ft_BERT
from src.fine_tune.GPT2 import Ft_GPT2
from src.dataset.dataloader import dataloader
from src.model.BERT import BERT
from src.model.GPT2 import GPT2

from peft import LoraConfig, TaskType, get_peft_model

class RpcClient:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.model_train = None
        self.train_loader = None
        self.model_name = None
        self.device = device
        self.fine_tune_config = None
        self.model = None

        self.response = None
        self.label_count = None

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]

        if action == "START":
            self.model_name = self.response["model_name"]
            state_dict = self.response["parameters"]
            cut_layers = self.response['cut_layers']
            total_block = self.response['total_block']
            self.fine_tune_config = self.response['fine_tune_config']
            bottleneck_config =self.response['bottleneck_config']
            bottleneck_state_dict = self.response["bottleneck"]

            if self.model_name == 'BERT':
                self.model_train = Ft_BERT(self.client_id, self.layer_id, self.channel, self.device)
                peft_config = LoraConfig(
                    task_type="SEQ_CLS",
                    r=self.fine_tune_config['LoRA']['r'], lora_alpha=self.fine_tune_config['LoRA']['alpha'],
                    lora_dropout=0.1,
                    bias="none",
                    target_modules=["query", "key", "value", "dense"])
                klass = BERT
            else:
                self.model_train = Ft_GPT2(self.client_id, self.layer_id, self.channel, self.device)
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.fine_tune_config['LoRA']['r'], lora_alpha=self.fine_tune_config['LoRA']['alpha'], lora_dropout=0.05,
                    bias="none",
                    target_modules=["c_attn", "c_proj", "c_fc"],
                    fan_in_fan_out=True
                )
                klass = GPT2

            self.model = None
            bottleneck_dict = {}

            if self.layer_id == 1:
                if bottleneck_config["enable"]:
                    self.model = klass(layer_id=1, n_block=cut_layers, reduce_comm=True,
                                  bottleneck_dim=bottleneck_config['bottleneck_dim'])
                    bottleneck_dict = {
                        k: v
                        for k, v in bottleneck_state_dict.items()
                        if k.startswith("encoder.")
                    }

                else:
                    self.model = klass(layer_id=1, n_block=cut_layers)
            if self.layer_id == 2:
                if bottleneck_config["enable"]:
                    self.model = klass(layer_id=2, n_block=total_block - cut_layers, reduce_comm=True,
                                  bottleneck_dim=bottleneck_config['bottleneck_dim'])
                    bottleneck_dict = {
                        k: v
                        for k, v in bottleneck_state_dict.items()
                        if k.startswith("decoder.")
                    }

                else:
                    self.model = klass(layer_id=2, n_block=total_block - cut_layers)

            # Read parameters and load to model
            if state_dict:
                state_dict = {**state_dict, **bottleneck_dict}
                self.model.load_state_dict(state_dict)

            if self.layer_id == 1:
                if self.fine_tune_config['client']:
                    self.model = get_peft_model(self.model, peft_config)
                    self.model.print_trainable_parameters()
                else:
                    for p in self.model.parameters():
                        p.requires_grad = False

            if self.layer_id == 2:
                if self.fine_tune_config['server']:
                    self.model = get_peft_model(self.model, peft_config)
                    self.model.print_trainable_parameters()
                else:
                    for p in self.model.parameters():
                        p.requires_grad = False
                if self.model_name == 'BERT':
                    for param in self.model.classifier.parameters():
                        param.requires_grad = True

            return True

        elif action == "SYN":
            label_counts = self.response['label_counts']
            stt = self.response['stt']
            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            weight_decay = self.response["weight_decay"]
            control_count = self.response["control_count"]

            # Start training
            if self.layer_id == 1:
                if self.train_loader is None:
                    src.Log.print_with_color(f"Label: {label_counts[stt]}", 'yellow')
                    self.train_loader = dataloader(self.model_name, batch_size, label_counts[stt], train=True)

                result, size = self.model_train.first_layer(self.model, lr, weight_decay,
                                                            control_count, self.train_loader)
            else:
                result, size = self.model_train.last_layer(self.model, lr, weight_decay)

            # Stop training, then send parameters to server
            if self.layer_id == 1:
                if self.fine_tune_config['client']:
                    self.model = self.model.merge_and_unload()
            else:
                if self.fine_tune_config['server']:
                    self.model = self.model.merge_and_unload()

            model_state_dict = copy.deepcopy(self.model.state_dict())

            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "result": result, "size": size,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True

        elif action == "STOP":
            return False

    def send_to_server(self, message):
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
