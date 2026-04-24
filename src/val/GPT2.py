import torch
import torch.nn as nn
from tqdm import tqdm

# import src.Log
from src.dataset.dataloader import dataloader
from src.model.GPT2 import GPT2

def val_GPT2(state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    test_loader = dataloader(model_name='GPT2', batch_size=4, distribution=[1000], train=False)

    model = GPT2()
    model.load_state_dict(state_dict_full)
    model = model.to(device)

    model.eval()

    total_loss  = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(input_ids, mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if torch.isnan(loss).any():
                print("NaN detected in loss")
                return False

            total_loss   += loss.item()

    avg_loss = total_loss / max(len(test_loader), 1)

    logger.log_info(f"Test Loss: {avg_loss:.4f}")

    return True