import os
import json
import random

from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
from datasets import load_dataset

from src.dataset.GSM8K import GSM8K
from src.dataset.CodeAlpaca import CodeAlpacaDataset
from src.dataset.EMOTION import EMOTIONDataset
from src.dataset.EMOTION import load_train_EMOTION
from src.dataset.EMOTION import load_test_EMOTION
from torch.utils.data import DataLoader

def dataloader(model_name =None, data_name=None, batch_size=None, distribution=500, train=True):
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

            train_set = GSM8K(tokenizer, train_set)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            path = os.path.join("data/", f"test.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]

            random.shuffle(data)

            test_set = data[:100]
            for ex in test_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(test_set)} test examples")
            test_set = GSM8K(tokenizer, test_set)
            test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
            return test_loader

    if data_name == 'CodeAlpaca':
        if model_name == 'GPT2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif model_name == 'Llama':
            tokenizer = AutoTokenizer.from_pretrained('JackFram/llama-160m')
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        ds = load_dataset(
            "HuggingFaceH4/CodeAlpaca_20K",
            download_mode="reuse_dataset_if_exist"
        )
        if train:
            train_list = list(ds['train'])

            random.shuffle(train_list)
            train_set = train_list[:distribution]

            for ex in train_set:
                ex['prompt'] = ex['prompt'] + "\n"
                eos = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
                ex['completion'] = ex['completion'] + eos

            print(f"CodeAlpaca: {len(train_set)} training examples")
            train_set = CodeAlpacaDataset(tokenizer, train_set, set_max_len=512, loss_on_prefix=False)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            test_list = list(ds['test'])

            random.shuffle(test_list)
            test_set = test_list[:100]

            for ex in test_set:
                ex['prompt'] = ex['prompt'] + "\n"
                eos = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
                ex['completion'] = ex['completion'] + eos

            print(f"CodeAlpaca: {len(test_set)} test examples")
            test_set = CodeAlpacaDataset(tokenizer, test_set, set_max_len=512, loss_on_prefix=False)
            test_loader = DataLoader(test_set, batch_size=4, shuffle=True)
            return test_loader

    if data_name == 'EMOTION':
        dataset = load_dataset(
            'ag_news',
            download_mode='reuse_dataset_if_exists',
            cache_dir='./hf_cache'
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if train:
            num_label = int(distribution / 4)
            distribution = [num_label, num_label, num_label, num_label]
            train_texts, train_labels = load_train_EMOTION(dataset, distribution)
            train_set = EMOTIONDataset(train_texts, train_labels, tokenizer, max_length=128)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            test_texts, test_label = load_test_EMOTION(2000, dataset)
            test_set = EMOTIONDataset(test_texts, test_label, tokenizer, max_length=128)
            test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
            return test_loader
