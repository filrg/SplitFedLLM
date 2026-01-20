import torch
import torch.nn as nn
from tqdm import tqdm

from src.model.Bert import Bert
from src.dataset.dataloader import dataloader

def val_Bert(state_dict_full, cut_layers , bottleneck_config, logger):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = dataloader(train=False)
    if bottleneck_config['enable']:
        client = Bert(layer_id=1, n_block=cut_layers, reduce_comm=True,
                                              bottleneck_dim=bottleneck_config['bottleneck_dim'])
        client.load_state_dict(state_dict_full[0])
        client = client.to(device)
        server = Bert(layer_id=2, n_block=12 - cut_layers, reduce_comm=True,
                                              bottleneck_dim=bottleneck_config['bottleneck_dim'])
        server.load_state_dict(state_dict_full[1])
        server = server.to(device)
    else:
        client = Bert(layer_id=1, n_block=cut_layers)
        client.load_state_dict(state_dict_full[0])
        client = client.to(device)
        server = Bert(layer_id=2, n_block= 12 - cut_layers)
        server.load_state_dict(state_dict_full[1])
        server = server.to(device)

    client.eval()
    server.eval()
    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = client(input_ids=input_ids)
            logits = server(input_ids=logits)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}; Test Acc: {acc:.4f}")

    logger.log_info(f"Test Loss: {avg_loss:.4f}; Test Acc: {acc:.4f}")









