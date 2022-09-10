# -*-coding:utf-8 -*-
import torch
import json


def data_loader(file_name):
    def helper():
        data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(json.loads(line.strip()))
        return data
    return helper


class SeqPairDataset():
    def __init__(self, data_loader, max_seq_len, tokenizer):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
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
            self.labels = [data['label'] for data in self.raw_data]

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label'] = self.labels[idx]
        return sample

    def __len__(self):
        return len(self.features)


class SeqPairMtlDataset():
    def __init__(self, data_loader, max_seq_len, tokenizer):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
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
        if 'label1' in self.raw_data[0]:
            self.labels = [{'label1': data['label1'], 'label2': data['label2']} for data in self.raw_data]

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label1'] = self.labels[idx]['label1']
            sample['label2'] = self.labels[idx]['label2']
        return sample

    def __len__(self):
        return len(self.features)