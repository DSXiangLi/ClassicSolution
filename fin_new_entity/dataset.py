# -*-coding:utf-8 -*-
import torch


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
