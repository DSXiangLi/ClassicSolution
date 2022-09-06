# -*-coding:utf-8 -*-
import torch

class SeqPairDataset():
    def __init__(self, max_seq_len, tokenizer, text1, text2=None, y=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.features = []
        self.labels = []
        self.build_feature(text1, text2, y)

    def build_feature(self, text1, text2, y):
        for t1, t2 in zip(text1, text2):
            self.features.append(self.tokenizer.encode_plus(t1, t2, padding='max_length',
                                                            truncation=True, max_length=self.max_seq_len, verbose=False))
        if y is not None:
            self.labels = [i for i in y ]

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label'] = self.labels[idx]
        return sample

    def __len__(self):
        return len(self.features)