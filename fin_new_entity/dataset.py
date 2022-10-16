# -*-coding:utf-8 -*-
import torch
import numpy as np


class SeqLabelDataset():
    def __init__(self, data_loader, tokenizer, max_seq_len, label2idx):
        self.raw_data = data_loader()
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
            feature = self.tokenizer.encode_plus(' '.join(data['text1']), padding='max_length',
                                                 is_split_into_words=True,  # split on white space
                                                 truncation=True, max_length=self.max_seq_len)
            self.features.append(feature)
            if self.has_label:
                # set CLS, SEP, PAD to 'O' in label, so that they won't impact transition
                label = [self.label2idx[i] for i in data['label'][:(self.max_seq_len - 2)]]
                label = [self.label2idx['O']] + label + [self.label2idx['O']]
                assert len(label) == sum(feature['attention_mask'])
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


class SpanDataset():
    def __init__(self, data_loader, tokenizer, max_seq_len):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.features = []
        self.labels = []
        self.build_feature()

    def build_feature(self):
        if self.raw_data[0].get('label'):
            self.has_label = True

        for data in self.raw_data:
            feature = self.tokenizer.encode_plus(' '.join(data['text1']), padding='max_length',
                                                 is_split_into_words=True,  # split on white space
                                                 truncation=True, max_length=self.max_seq_len)
            self.features.append(feature)
            if self.has_label:
                # set CLS, SEP, PAD to 'O' in label, so that they won't impact transition
                label_start = data['label'][:(self.max_seq_len - 2)]
                label_end = data['label2'][:(self.max_seq_len - 2)]
                label_start = [0] + label_start + [0]
                label_end = [0] + label_end + [0]
                assert len(label_start) == sum(feature['attention_mask'])
                assert len(label_end) == sum(feature['attention_mask'])
                label_start += [0] * (self.max_seq_len - len(label_start))
                label_end += [0] * (self.max_seq_len - len(label_end))
                self.labels.append({'label_start': label_start, 'label_end': label_end})

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label_start'] = torch.tensor(self.labels[idx]['label_start'])
            sample['label_end'] = torch.tensor(self.labels[idx]['label_end'])
        return sample

    def __len__(self):
        return len(self.features)


class GlobalPointerDataset():
    def __init__(self, data_loader, tokenizer, max_seq_len, num_head):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.num_head = num_head
        self.features = []
        self.labels = []
        self.build_feature()

    def build_feature(self):
        if self.raw_data[0].get('label'):
            self.has_label = True

        for data in self.raw_data:
            feature = self.tokenizer.encode_plus(' '.join(data['text1']), padding='max_length',
                                                 is_split_into_words=True,  # split on white space
                                                 truncation=True, max_length=self.max_seq_len)
            self.features.append(feature)
            if self.has_label:
                label_pointer = np.zeros((self.num_head, self.max_seq_len, self.max_seq_len))
                # pos+1 for [CLS] at the beginning
                for pos in data['label']:
                    if pos[2] + 1 >= self.max_seq_len:
                        continue
                    label_pointer[pos[0], pos[1] + 1, pos[2] + 1] = 1

                self.labels.append(label_pointer)

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label_ids'] = torch.tensor(self.labels[idx])
        return sample

    def __len__(self):
        return len(self.features)
