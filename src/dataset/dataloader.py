import torch
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

import random
from collections import defaultdict

from src.dataset.SQUAD import SQUAD_DATASET
from src.dataset.AGNEWS import AGNEWS_DATASET
from torch.utils.data import DataLoader

def AGNEWS(batch_size=None, distribution=None, train=True):
    cache_dir = './hf_cache'
    print(f"Loading AGNEWS dataset with cache_dir={cache_dir}...")
    dataset = load_dataset(
        'ag_news',
        download_mode='reuse_dataset_if_exists',
        cache_dir=cache_dir
    )
    print("Dataset loaded successfully.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    if train:
        train_data = dataset['train']
        train_target_counts = {k: v for k, v in enumerate(distribution)}
        train_by_class = defaultdict(list)
        for text, label in zip(train_data['text'], train_data['label']):
            train_by_class[label].append((text, label))

        train_texts, train_labels = [], []
        for label, count in train_target_counts.items():
            samples = random.sample(train_by_class[label], count)
            train_texts.extend([t for t, _ in samples])
            train_labels.extend([l for _, l in samples])
        print("Train samples:", len(train_texts), {l: train_labels.count(l) for l in set(train_labels)})

        train_set = AGNEWS_DATASET(train_texts, train_labels, tokenizer, max_length=128)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        test_data = dataset['test']
        distribution = [500, 500, 500, 500]
        test_target_counts = {k: v for k, v in enumerate(distribution)}
        test_by_class = defaultdict(list)
        for text, label in zip(test_data['text'], test_data['label']):
            test_by_class[label].append((text, label))

        test_texts, test_labels = [], []
        for label, count in test_target_counts.items():
            samples = random.sample(test_by_class[label], count)
            test_texts.extend([t for t, _ in samples])
            test_labels.extend([l for _, l in samples])

        print("Test samples:", len(test_texts), {l: test_labels.count(l) for l in set(test_labels)})

        test_set = AGNEWS_DATASET(test_texts, test_labels, tokenizer, max_length=128)
        test_loader = DataLoader(test_set, batch_size=50, shuffle=False)
        return test_loader

def SQUAD(batch_size, distribution=None, train=True):

    if distribution is None:
        distribution = [2000]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if train:
        train_data = torch.load("./data/SQUAD/TRAIN_SQUAD.pt")

        random.seed(42)
        subset = random.sample(train_data, distribution[0])

        train_set = SQUAD_DATASET(tokenizer, subset, 256)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        return train_loader
    else:
        test_data = torch.load("./data/SQUAD/VAL_SQUAD.pt")

        test_set = SQUAD_DATASET(tokenizer, test_data[:distribution[0]], 256)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return test_loader

def dataloader(model_name, batch_size=None, distribution=None, train=True):
    if model_name == 'BERT':
        data = AGNEWS(batch_size, distribution, train)
    elif model_name == 'GPT2':
        data = SQUAD(batch_size, distribution, train)
    else:
        raise ValueError(f"Dataset of model {model_name} not supported.")

    return data
