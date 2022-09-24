# -*-coding:utf-8 -*-
from collections import Counter
from functools import partial
import numpy as np
import torch

from collections import defaultdict


def extract_entity(text, pos_list):
    l = len(text)
    ent = defaultdict(set)
    for pos in pos_list:
        # allow pos list to be longer than text
        if pos[1]>=l:
            continue
        ent[pos[0]].add(text[pos[1]: pos[2]])
    return ent


def get_entity_bio(tags, idx2label):
    """
    Input:
        tags: list of labels [O,O,O, B-FIN, I-FIN, O,O, B-LOC,I-LOC]
    Return:
        type of span with position, [left, right)
        [['FIN',3,5], ['LOC',7,9]]
    """
    tags = [idx2label[i] for i in tags]
    span = []
    type1 = ''
    pos = [-1, -1]
    for i, tag in enumerate(tags):
        if 'B' in tag:
            if pos[1] != -1:
                span.append([type1] + pos)
            type1 = tag.split('-')[1]
            pos = [i, i + 1]
        elif 'I' in tag and pos[0] != -1:
            if tag.split('-')[1] == type1:
                pos[1] = i + 1
        else:
            if type1:
                span.append([type1] + pos)
                type1 = ''
                pos = [-1, -1]
    if type1:
        span.append([type1] + pos)
    return span


def get_spans(tags, idx2label, schema):
    if schema == 'BIO':
        return get_entity_bio(tags, idx2label)
    else:
        raise ValueError('Only BIO tagging schema is supported now')


class SpanMetricBase(object):
    def __init__(self, idx2label, schema='BIO', avg='micro'):
        assert avg in ['micro', 'macro'], 'Only [micro, macro] are supported'
        self.get_spans = partial(get_spans, idx2label=idx2label, schema=schema)
        self.avg = avg
        self.true_span = []
        self.pred_span = []
        self.right_span = []
        self.class_info = {}

    def update(self, pred_ids, label_ids):
        '''
        Input: list of label_ids and pred_ids with real seq length
            labels_ids_list = [['O','B-FIN', 'I-FIN', 'O'], ['B-LOC', 'I-LOC', 'O']]
            pred_ids_list  = [['O','B-FIN', 'I-FIN', 'O'], ['B-LOC', 'I-LOC', 'O']]
        '''

        label_spans = self.get_spans(label_ids)
        pred_spans = self.get_spans(pred_ids)
        self.true_span.extend(label_spans)
        self.pred_span.extend(pred_spans)
        self.right_span.extend([i for i in pred_spans if i in label_spans])

    def get_detail(self):
        return self.class_info


class SpanPrecision(SpanMetricBase):
    def __init__(self, idx2label, schema='BIO', avg='micro'):
        super(SpanPrecision, self).__init__(idx2label, schema, avg)

    def compute(self):
        pred_counter = Counter([x[0] for x in self.pred_span])
        right_counter = Counter([x[0] for x in self.right_span])
        for key, val in pred_counter.items():
            self.class_info[key] = right_counter.get(key, 0) / pred_counter[key]
        if self.avg == 'micro':
            precision = len(self.right_span) / len(self.pred_span)
        else:
            precision = np.mean(list(self.class_info.values()))
        return torch.tensor(precision)


class SpanRecall(SpanMetricBase):
    def __init__(self, idx2label, schema='BIO', avg='micro'):
        super(SpanRecall, self).__init__(idx2label, schema, avg)

    def compute(self):
        true_counter = Counter([x[0] for x in self.true_span])
        right_counter = Counter([x[0] for x in self.right_span])
        for key, val in true_counter.items():
            self.class_info[key] = right_counter.get(key, 0) / true_counter[key]
        if self.avg == 'micro':
            recall = len(self.right_span) / len(self.true_span)
        else:
            recall = np.mean(list(self.class_info.values()))
        return torch.tensor(recall)

    def get_detail(self):
        return self.class_info


class SpanF1(SpanMetricBase):
    def __init__(self, idx2label, schema='BIO', avg='micro'):
        super(SpanF1, self).__init__(idx2label, schema, avg)
        self.epsilon = 1e-10

    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0
        else:
            return (2 * precision * recall) / (precision + recall)

    def compute(self):
        pred_counter = Counter([x[0] for x in self.pred_span])
        true_counter = Counter([x[0] for x in self.true_span])
        right_counter = Counter([x[0] for x in self.right_span])
        for key in set(list(pred_counter.keys()) + list(true_counter.keys())):
            recall = right_counter[key] / true_counter[key] if key in true_counter else 0
            precision = right_counter[key] / pred_counter[key] if key in pred_counter else 0
            self.class_info[key] = self.f1_score(precision, recall)

        if self.avg == 'micro':
            recall = len(self.right_span) / len(self.true_span)
            precision = len(self.right_span) / len(self.pred_span)
            f1 = self.f1_score(precision, recall)
        else:
            f1 = np.mean(list(self.class_info.values()))
        return torch.tensor(f1)

    def get_detail(self):
        return self.class_info


def pad_sequence(input_, pad_len=None, pad_value=0):
    """
    Pad List[List] sequence to same length
    """
    output = []
    for i in input_:
        output.append(i + [pad_value] * (pad_len - len(i)))
    return output


if __name__ == '__main__':
    label2idx = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4}
    idx2label = {j: i for i, j in label2idx.items()}
    print(get_entity_bio([0, 1, 2, 2], idx2label))
    print(get_entity_bio([0, 1, 2, 1, 2], idx2label))
    print(get_entity_bio([0, 1, 2, 0, 1, 2, 0], idx2label))
    print(get_entity_bio([1, 2, 0, 3, 4, 0], idx2label))
    print(get_entity_bio([1, 2, 3, 4], idx2label))
    print(get_entity_bio([1, 1], idx2label))
    print(get_entity_bio([0, 2, 2, 0], idx2label))
    print(get_entity_bio([0, 1, 4, 0], idx2label))

    metrics = {}
    for avg in ['micro', 'macro']:
        metrics.update({
            f'f1_{avg}': SpanF1(idx2label=idx2label, schema='BIO', avg=avg),
            f'recall_{avg}': SpanRecall(idx2label=idx2label, schema='BIO', avg=avg),
            f'precision_{avg}': SpanPrecision(idx2label=idx2label, schema='BIO', avg=avg)
        })

    preds = [[0, 1, 2, 0, 1, 2, 0, 0]]
    label_ids = [[0, 1, 2, 0, 1, 2, 0, 0]]
    for metric in metrics.values():
        metric.update(preds, label_ids)
    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    print(multi_metrics)

    preds = [[0, 1, 2, 0, 1, 2, 3, 4]]
    label_ids = [[0, 1, 2, 0, 1, 2, 0, 0]]
    for metric in metrics.values():
        metric.update(preds, label_ids)
    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    print(multi_metrics)
