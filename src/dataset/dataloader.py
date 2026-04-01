import os
import json
import random

from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
from datasets import load_dataset

from src.dataset.GSM8K import GSM8K
from src.dataset.EMOTION import EMOTIONDataset
from src.dataset.EMOTION import load_train_EMOTION
from src.dataset.EMOTION import load_test_EMOTION
from torch.utils.data import DataLoader

def dataloader(model_name =None, data_name=None, batch_size=None, distribution=500, train=True):
    if data_name == 'E2E':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        path = os.path.join("data/", "e2e_train.jsonl")

        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        random.shuffle(data)

        from torch.utils.data import Dataset

        class E2EDataset(Dataset):
            def __init__(self, tokenizer, data, max_length=512):
                self.tokenizer = tokenizer
                self.data = data
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]

                prompt = f"<MR> {ex['input']} </MR> <TEXT>"
                target = ex["output"] + " <|endoftext|>"

                full = prompt + " " + target

                enc = self.tokenizer(
                    full,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)

                labels = input_ids.clone()


                prompt_len = len(
                    self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                )
                labels[:prompt_len] = -100

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }

        train_set = E2EDataset(tokenizer, data)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        return train_loader
    if data_name == 'GSM8K':
        if model_name == 'GPT2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif model_name == 'Llama':
            tokenizer = AutoTokenizer.from_pretrained('JackFram/llama-160m')
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if train:
            path = os.path.join("data/", f"train.jsonl")

            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]

            random.shuffle(data)

            train_set = data[:distribution]
            for ex in train_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(train_set)} train examples")

            from torch.utils.data import Dataset

            class E2EDataset(Dataset):
                def __init__(self, tokenizer, data, max_length=512):
                    self.tokenizer = tokenizer
                    self.data = data
                    self.max_length = max_length

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    ex = self.data[idx]

                    prompt = f"### Input:\n{ex['question']}\n\n### Output:\n"
                    target = ex["answer"] + " <|endoftext|>"

                    full = prompt + target

                    enc = self.tokenizer(
                        full,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )

                    input_ids = enc["input_ids"].squeeze(0)
                    attention_mask = enc["attention_mask"].squeeze(0)


                    labels = input_ids.clone()

                    prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
                    prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

                    labels[:prompt_len] = -100  

                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
            train_set = E2EDataset(tokenizer, train_set)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            path = os.path.join("data/", f"test.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]

            random.shuffle(data)

            test_set = data[:500]
            for ex in test_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(test_set)} test examples")
            test_set = GSM8K(tokenizer, test_set,False)
            test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
            return test_loader

    if data_name == 'EMOTION':

        dataset = load_dataset(
            'emotion',
            download_mode='reuse_dataset_if_exists',
            cache_dir='./hf_cache'
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        if train:
            # EMOTION có 6 class (sadness/joy/love/anger/fear/surprise)
            num_label = int(distribution / 6)
            distribution = [num_label] * 6
            train_texts, train_labels = load_train_EMOTION(dataset, distribution)
            train_set = EMOTIONDataset(train_texts, train_labels, tokenizer, max_length=128)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            test_texts, test_label = load_test_EMOTION(2000, dataset)
            test_set = EMOTIONDataset(test_texts, test_label, tokenizer, max_length=128)
            test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
            return test_loader

    elif data_name == 'AG_NEWS':
        dataset = load_dataset(
            'ag_news',
            download_mode='reuse_dataset_if_exists',
            cache_dir='./hf_cache'
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        if train:
            num_label = int(distribution / 4)
            distribution = [num_label] * 4
            train_texts, train_labels = load_train_EMOTION(dataset, distribution)
            train_set = EMOTIONDataset(train_texts, train_labels, tokenizer, max_length=128)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            test_texts, test_label = load_test_EMOTION(2000, dataset)
            test_set = EMOTIONDataset(test_texts, test_label, tokenizer, max_length=128)
            test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
            return test_loader