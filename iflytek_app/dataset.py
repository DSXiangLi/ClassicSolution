# -*-coding:utf-8 -*-

import torch

class MixDataset():
    """
    字符 + 词粒度 Dataset
    """
    def __init__(self, max_seq_len, w2v, c2v, phraser, label2idx, text1, text2=None, y=None):
        self.max_seq_len = max_seq_len
        self.features = []
        self.labels = []
        self.w2v = w2v
        self.c2v = c2v
        self.phraser =phraser
        self.label2idx = label2idx
        self.build_feature(text1, text2, y)

    def build_feature(self, text1, text2, y):
        for i, (a, b) in enumerate(zip(text1, text2)):
            # token 粒度
            tokens = a + ['[SEP]'] + b
            tokens = tokens[:self.max_seq_len]

            seq_len = len(tokens)
            attention_mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

            tokens += ['[PAD]'] * (self.max_seq_len - seq_len)
            token_ids = self.c2v.convert_tokens_to_ids(tokens)
            ## word 粒度
            words = self.phraser[tokens]
            word_ids = []
            for word in words:
                word_ids += self.w2v.convert_tokens_to_ids([word]) * len(word.split('_'))
            self.features.append({'token_ids': token_ids,
                                  'word_ids': word_ids,
                                  'attention_mask': attention_mask,
                                  'seq_len': seq_len,
                                  'idx': i})

        if y is not None:
            for i in y:
                self.labels.append(self.label2idx[i])

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label'] = self.labels[idx]
        return sample

    def __len__(self):
        return len(self.features)
