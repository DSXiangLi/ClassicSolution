import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from src.seqlabel_utils import pos2bio


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


class SeqLabelDataset():
    """
    Dataset for Sequence Labelling, by default only take single text input
    """

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
                label = pos2bio(data['text1'], data['label'])
                # set CLS, SEP, PAD to 'O' in label, so that they won't impact transition
                label = [self.label2idx[i] for i in label[:(self.max_seq_len - 2)]]
                label = [self.label2idx['O']] + label + [self.label2idx['O']]
                assert len(label) == sum(feature['attention_mask'])
                label += [self.label2idx['O']] * (self.max_seq_len - len(label))
                self.labels.append(label)

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label_ids'] = torch.tensor(self.labels[idx], dtype=torch.int32)
        return sample

    def __len__(self):
        return len(self.raw_data)



class SpanDatasetG(Dataset):
    """
    Flatten：直接用多标签指针网络预测事件Argument
    """

    def __init__(self, data_loader, tokenizer, max_seq_len, label2idx):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.features = self.build_feature()

    def build_feature(self):
        if self.raw_data[0].get('label'):
            self.has_label = True

        features = {}
        for data in self.raw_data:
            feature = self.tokenizer.encode_plus(' '.join(data['text1']), padding='max_length',
                                                 is_split_into_words=True,  # split on white space
                                                 truncation=True, max_length=self.max_seq_len)
            if self.has_label:
                ## 序列多标签
                label_start = [np.zeros(len(self.label2idx), dtype=np.float32) for i in range(self.max_seq_len)]
                label_end = [np.zeros(len(self.label2idx), dtype=np.float32) for i in range(self.max_seq_len)]
                for l in data['label']:
                    if l[2] < (self.max_seq_len - 2):
                        label_start[l[1]][self.label2idx[l[0]]] = 1
                        label_end[l[2]][self.label2idx[l[0]]] = 1
                    else:
                        break
                feature.update({'label_start': label_start.copy(), 'label_end': label_end.copy()})
                yield feature

    def __getitem__(self, idx):
        sample = next(self.features)
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        return sample

    def __len__(self):
        return len(self.raw_data)


class EventSpanDataset(Dataset):
    """
    Joint Event Extraction：同时预测Argument多标签span，以及多标签事件分类
    """
    def __init__(self, data_loader, tokenizer, max_seq_len, label2idx):
        self.raw_data = data_loader()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.features = []
        self.labels = []
        self.label2idx = label2idx
        self.feature = self.build_feature()

    def build_feature(self):
        if self.raw_data[0].get('label'):
            self.has_label = True

        for data in self.raw_data:
            feature = self.tokenizer.encode_plus(' '.join(data['text1']), padding='max_length',
                                                 is_split_into_words=True,  # split on white space
                                                 truncation=True, max_length=self.max_seq_len)
            self.features.append(feature)
            if self.has_label:
                label_start = [0] * self.max_seq_len
                label_end = [0] * self.max_seq_len
                for l in data['label']:
                    if l[2] < self.max_seq_len - 2:
                        label_start[l[1]] = self.label2idx[l[0]]
                        label_end[l[2]] = self.label2idx[l[0]]
                    else:
                        break
                self.labels.append({'label_start': label_start, 'label_end': label_end,
                                    'label_event': self.gen_multilabel(data['label2'])})

    def gen_multilabel(self, labels):
        one_hot = np.zeros(len(self.label2idx))
        for l in labels:
            one_hot[self.label2idx[l]] = 1
        return one_hot

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label_start'] = torch.tensor(self.labels[idx]['label_start'])
            sample['label_end'] = torch.tensor(self.labels[idx]['label_end'])
            sample['label_event'] = torch.tensor(self.labels[idx]['label_event'])
        return sample

    def __len__(self):
        return len(self.features)