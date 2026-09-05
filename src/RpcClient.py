import time
import pickle
import copy

import src.Log
from src.fine_tune.scheduler import create_scheduler
from src.dataset.dataloader import dataloader
from src.model.GPT2 import GPT2
from src.model.Llama import Llama
from src.model.Bert import Bert

from peft import LoraConfig, TaskType, get_peft_model

class RpcClient:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.model_train = None
        self.train_loader = None
        self.device = device

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
        state_dict = self.response["parameters"]

        if action == "START":
            model = None
            model_name = self.response["model_name"]
            cut_layers = self.response['cut_layers']
            # label_count = self.response['label_count']
            total_block = self.response['total_block']
            clip_grad_norm = self.response['clip_grad_norm']
            data_name = self.response["data_name"]
            num_sample = self.response["num_sample"]
            fine_tune_config = self.response['fine_tune_config']

            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            weight_decay = self.response["weight_decay"]
            control_count = self.response["control_count"]

            self.model_train = create_scheduler(
                model_name, self.client_id, self.layer_id, self.channel, self.device
            )

            if fine_tune_config['name'] == 'LoRA':
                if model_name == 'GPT2':
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=fine_tune_config['LoRA']['r'], lora_alpha=fine_tune_config['LoRA']['alpha'], lora_dropout=0.05, bias="none",
                        target_modules=["c_attn", "c_proj", "c_fc"],
                        fan_in_fan_out=True
                    )
                elif model_name == 'Llama':
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=fine_tune_config['LoRA']['r'], lora_alpha=fine_tune_config['LoRA']['alpha'], lora_dropout=0.05, bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
                elif model_name == 'Bert':
                    peft_config = LoraConfig(
                        task_type="SEQ_CLS",
                        r=8,lora_alpha=16,lora_dropout=0.1,bias="none",
                        target_modules=["query", "key", "value", "dense"]
                    )
                else:
                    peft_config = None
            else:
                peft_config = None

            if model_name == 'GPT2':
                klass = GPT2
            elif model_name == 'Llama':
                klass = Llama
            elif model_name == 'Bert':
                klass = Bert
            else:
                klass = globals()[f'GPT2']

            if self.layer_id == 1:
                model = klass(layer_id=1, n_block=cut_layers)

            else:
                model = klass(layer_id=2, n_block=total_block-cut_layers)

            # Read parameters and load to model
            if state_dict:
                model.load_state_dict(state_dict)

            if fine_tune_config['enable']:
                if model_name == 'Bert':
                    model = get_peft_model(model, peft_config)
                    if self.layer_id == 2:
                        for param in model.classifier.parameters():
                            param.requires_grad = True
                else:
                    model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            model.to(self.device)

            # Start training
            if self.layer_id == 1:
                if self.train_loader is None:
                    self.train_loader = dataloader(model_name, data_name, batch_size, num_sample, train=True)

                result, size = self.model_train.first_layer(model, lr, weight_decay, clip_grad_norm,
                                                                         control_count, self.train_loader)

            else:
                result, size = self.model_train.last_layer(model, lr, weight_decay, clip_grad_norm)

            # Stop training, then send parameters to server
            if fine_tune_config['enable']:
                model = model.merge_and_unload()

            model_state_dict = copy.deepcopy(model.state_dict())

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
