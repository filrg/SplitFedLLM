import pika
import uuid
import argparse
import yaml
import os
import math
import socket

import torch

import src.Log
from src.RpcClient import RpcClient

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--compute-weight', type=float, default=1.0, help='Relative compute capacity (positive); workers adapt after measured windows')
parser.add_argument('--tensor-host', help='Ethernet IP advertised to peers; inferred from route to broker if omitted')
parser.add_argument('--tensor-bind', default='0.0.0.0', help='Local TCP tensor listener address')
parser.add_argument('--tensor-port', type=int, default=0, help='TCP tensor port; 0 chooses a free port')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

device = None
if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
channel = connection.channel()

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")

    if not math.isfinite(args.compute_weight) or args.compute_weight <= 0:
        raise ValueError('--compute-weight must be finite and positive')
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id,
            "compute_weight": args.compute_weight, "message": "Hello from Client!"}
    transport = None
    options = config['learning'].get('u-shape', {})
    try:
        if config['server'].get('architecture', 'u-shape') == 'u-shape' and options.get('transport', 'rabbitmq') == 'tcp':
            from src.transport.tcp import TcpTensorTransport
            host = args.tensor_host
            if not host:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                    route.connect((address, 5672))  # Route lookup, no application data sent.
                    host = route.getsockname()[0]
            transport = TcpTensorTransport(
                client_id, device, args.tensor_bind, args.tensor_port,
                pool_bytes=options.get('pool-bytes', 64 * 1024 * 1024),
                pinned=options.get('pinned-memory', 'auto'),
                queue_size=options.get('transport-queue-size', 32),
                timeout=options.get('timeout-seconds', 120))
            data['tensor_endpoint'] = dict(host=host, port=transport.port)
            print('TCP tensor endpoint:', data['tensor_endpoint'])
        client = RpcClient(client_id, args.layer_id, channel, device, tensor_transport=transport)
        client.send_to_server(data)
        client.wait_response()
    finally:
        if transport is not None:
            transport.close()
        connection.close()

