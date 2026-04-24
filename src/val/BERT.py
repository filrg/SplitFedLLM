import torch
import torch.nn as nn
from tqdm import tqdm

from src.model.BERT import BERT
from src.dataset.dataloader import dataloader

def val_BERT(state_dict_full, logger):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = dataloader(model_name='BERT', batch_size=4 ,distribution=[1000] ,train=False)

    model = BERT()
    model.load_state_dict(state_dict_full)
    model = model.to(device)

    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids)
            loss = criterion(logits, labels)
            if torch.isnan(loss).any():
                return False
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}; Test Acc: {acc:.4f}")

    logger.log_info(f"Test Loss: {avg_loss:.4f}; Test Acc: {acc:.4f}")
    return True









