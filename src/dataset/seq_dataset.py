# -*-coding:utf-8 -*-
import torch
from torch.utils.data.dataset import Dataset


class SeqDataset(Dataset):
    def __init__(self, file_name, max_len, tokenizer, data_loader):
        self.raw_data = data_loader(file_name)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.features = []
        self.labels = []
        self.build_feature()

    def build_feature(self):
        for data in self.raw_data:
            self.features.append(self.tokenizer.encode_plus(data['text1'], padding='max_length',
                                                            truncation=True, max_length=self.max_len))
        if self.raw_data[0].get('label'):
            self.labels = [i['label'] for i in self.raw_data]

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label'] = self.labels[idx]
        return sample

    def __len__(self):
        return len(self.raw_data)


class SeqLabelDataset(Dataset):
    def __init__(self, file_name, max_seq_len, tokenizer, data_loader, label2idx):
        self.raw_data = data_loader(file_name)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.features = []
        self.labels = []
        self.label2idx = label2idx
        self.has_label = False
        self.build_feature()

    def build_feature(self):
        if self.raw_data[0].get('label'):
            self.has_label = True

        for data in self.raw_data:
            feature = self.tokenizer.encode_plus(data['text1'], padding='max_length',
                                                 truncation=True, max_length=self.max_seq_len)
            self.features.append(feature)
            if self.has_label:
                # set CLS, SEP, PAD to 'O' in label, so that they won't impact transition
                label = data['label'][:(self.max_seq_len - 2)]
                label = [self.label2idx['O']] + label + [self.label2idx['O']]
                label += [self.label2idx['O']] * (self.max_seq_len - len(label))
                self.labels.append(label)

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label_ids'] = torch.tensor(self.labels[idx])
        return sample

    def __len__(self):
        return len(self.raw_data)