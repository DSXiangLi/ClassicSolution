# -*-coding:utf-8 -*-
import torch
import json
from torch.utils.data import Dataset
from src.dataset.converter import data_loader


class SeqMlmDataset(Dataset):
    def __init__(self, data_loader, max_seq_len, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        self.raw_data = data_loadergit()
        self.max_seq_len = max_seq_len
        self.build_feature()

    def build_feature(self):
        for data in self.raw_data:
            input_ids = self.tokenizer.encode_plus(data['text1'], truncation=True, max_length=self.max_seq_len).input_ids
            ref_ids = data['ref']
            self.examples.append({'input_ids': input_ids, 'chinese_ref': ref_ids})

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

class SeqPairDataset(Dataset):
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


class SeqPairMtlDataset(Dataset):
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