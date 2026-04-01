import pika
import torch

from requests.auth import HTTPBasicAuth
import requests


def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue") or queue_name.startswith("rpc_queue"):

                http_channel.queue_delete(queue=queue_name)

            else:
                http_channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    else:
        return False

def change_keys(state_dict, num, increase=True):
    exclude_prefix = ["h.", "layers."]
    new_state_dict = {}
    for k, v in state_dict.items():
        if any(k.startswith(prefix) for prefix in exclude_prefix):
            parts = k.split(".")
            if increase:
                parts[1] = str(int(parts[1]) + num)
            else:
                parts[1] = str(int(parts[1]) - num)
            new_key = ".".join(parts)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            continue

    return new_state_dict

def fed_avg_state_dicts(state_dicts, weights=None):
    num = len(state_dicts)
    if num == 0:
        raise ValueError("fed_avg_state_dicts: don't have any state_dict.")

    if weights is None:
        weights = [1.0] * num
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("fed_avg_state_dicts: tổng weight = 0.")

    all_keys = set().union(*(sd.keys() for sd in state_dicts))
    avg_dict = {}

    for key in all_keys:
        acc   = None
        acc_w = 0.0   # Fix: theo dõi tổng weight thực sự đóng góp cho key này

        for sd, w in zip(state_dicts, weights):
            if key not in sd:
                continue
            t = sd[key].float()
            if torch.isnan(t).any():
                t = torch.nan_to_num(t, nan=0.0)
            t = t * w
            acc    = t if acc is None else acc + t
            acc_w += w

        # Fix: nếu key không có trong bất kỳ state_dict nào → bỏ qua
        if acc is None or acc_w == 0:
            continue

        avg = acc / acc_w   # Fix: chia cho tổng weight thực tế của key, không total_w

        orig = next((sd[key] for sd in state_dicts if key in sd), None)
        if orig is None:
            continue
        if orig.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            avg = avg.round().to(orig.dtype)
        else:
            avg = avg.to(orig.dtype)

        avg_dict[key] = avg

    return avg_dict