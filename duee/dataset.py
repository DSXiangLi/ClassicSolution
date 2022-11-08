import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class SeqMultiLabelDataset(Dataset):
    def __init__(self, data_loader, tokenizer, max_seq_len, label2idx):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.features = []
        self.labels = []
        self.build_feature()

    def build_feature(self):
        if 'text2' in self.raw_data[0]:
            for data in self.raw_data:
                self.features.append(self.tokenizer.encode_plus(data['text1'], data['text2'], padding='max_length',
                                                                truncation=True, max_length=self.max_seq_len,
                                                                verbose=False))
        else:
            for data in self.raw_data:
                self.features.append(self.tokenizer.encode_plus(data['text1'], padding='max_length',
                                                                truncation=True, max_length=self.max_seq_len,
                                                                verbose=False))
        if 'label' in self.raw_data[0]:
            self.labels = [self.gen_multilabel(data['label']) for data in self.raw_data]

    def gen_multilabel(self, labels):
        one_hot = np.zeros(len(self.label2idx))
        for l in labels:
            one_hot[self.label2idx[l]] = 1
        return one_hot

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label'] = torch.tensor(self.labels[idx])
        return sample

    def __len__(self):
        return len(self.features)

