from transformers import BertTokenizer
from datasets import load_dataset

from src.dataset.EMOTION import EMOTIONDataset
from src.dataset.EMOTION import load_train_EMOTION
from src.dataset.EMOTION import load_test_EMOTION
from torch.utils.data import DataLoader

def dataloader(batch_size=None, distribution=None, train=True):
    dataset = load_dataset(
        'ag_news',
        download_mode='reuse_dataset_if_exists',
        cache_dir='./hf_cache'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    if train:
        train_texts, train_labels = load_train_EMOTION(dataset, distribution)
        train_set = EMOTIONDataset(train_texts, train_labels, tokenizer, max_length=128)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        test_texts, test_label = load_test_EMOTION(2000, dataset)
        test_set = EMOTIONDataset(test_texts, test_label, tokenizer, max_length=128)
        test_loader = DataLoader(test_set, batch_size=25, shuffle=False)
        return test_loader
