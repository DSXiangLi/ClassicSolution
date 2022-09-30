# -*-coding:utf-8 -*-
"""
    序列标注，Span标注，全局指针的序列抽取问题都依赖 position list来完成实体到label的转换
    pos_list [实体类别，左闭，右闭]:[[LOC, 2,3], [PER, 3,5]]
"""
from collections import Counter
from functools import partial
import numpy as np
import torch

from collections import defaultdict


def pos2bio(text, pos_list):
    """
    Input:
        text: 文本
        pos_list: [[FIN, 0,3], [LOC, 7,8]]
    """
    label = ['O'] * len(text)
    for pos in pos_list:
        label[pos[1]] = 'B-' + pos[0]
        label[(pos[1] + 1):(pos[2] + 1)] = ['I-' + pos[0]] * (pos[2] - pos[1])
    return label


def pos2span(text, pos_list, type2idx):
    """
    Input:
        text: 文本
        pos_list: [[FIN, 0,3], [LOC, 7,8]]
        type2idx: {FIN:1, LOC:2, PER:3} id start from 1
    """
    start_label = [0] * len(text)
    end_label = [0] * len(text)
    for pos in pos_list:
        start_label[pos[1]] = type2idx[pos[0]]
        end_label[pos[2]] = type2idx[pos[0]]
    return start_label, end_label


def pos2pointer(pos_list, type2idx):
    """
    Input:
        text: 文本
        pos_list: [[FIN, 0,3], [LOC, 7,8]]
        type2idx: {FIN:0, LOC:1, PER:2} id start from 0
    """
    label = []
    for pos in pos_list:
        label.append([type2idx[pos[0]], pos[1], pos[2]])
    return label


def extract_entity(text, pos_list):
    l = len(text)
    ent = defaultdict(set)
    for pos in pos_list:
        # allow pos list to be longer than text
        if pos[1] >= l:
            continue
        ent[pos[0]].add(text[pos[1]: (pos[2] + 1)])
    return ent


def get_entity_bio(tags, idx2label=None):
    """
    Input:
        tags: list of labels or label_ids [O,O,O, B-FIN, I-FIN, O,O, B-LOC,I-LOC],[0,0,0,1,2,0,0,3,4], where CLS is removed
        idx2label： 如果为None，默认传入的是labels, 否则传入的是的label_ids
    Return:
        pos list: [['FIN',3,5], ['LOC',7,9]]
    """
    if idx2label is not None:
        tags = [idx2label[i] for i in tags]
    pos_list = []
    type1 = ''
    pos = [-1, -1]
    for i, tag in enumerate(tags):
        if 'B' in tag:
            if pos[1] != -1:
                pos_list.append([type1] + pos)
            type1 = tag.split('-')[1]
            pos = [i, i]
        elif 'I' in tag and pos[0] != -1:
            if tag.split('-')[1] == type1:
                pos[1] = i
        else:
            if type1:
                pos_list.append([type1] + pos)
                type1 = ''
                pos = [-1, -1]
    if type1:
        pos_list.append([type1] + pos)
    return pos_list


def get_entity_span(tags_pair, idx2label, max_span=20):
    """
    Input:
        tags_pair: [pos_start_list, pos_end_list], where CLS is removed
        idx2label: {1: 'LOC',2:'PER'}
        max_search: span的最大长度是20，
    Return:
        pos list: [['FIN',3,5], ['LOC',7,9]]
    """
    start_pos = tags_pair[0]
    end_pos = tags_pair[1]
    pos_list = []
    l = len(start_pos)
    for i, s in enumerate(start_pos):
        if s == 0:
            continue
        for j in range(i, i + max_span):
            if j >= l:
                break
            if end_pos[j] == s:
                pos_list.append([idx2label[s], i, j])
                break
    return pos_list


def get_entity_pointer(tags_matrix, idx2label):
    """
    Input:
        tags_matrix: (i,j) =id， id类型[i,j]之间维实体, where CLS isremoved
        idx2label: {1: 'LOC',2:'PER'}
    Return:
        pos list: [['FIN',3,5], ['LOC',7,9]]
    """
    l = len(tags_matrix)
    pos_list = []
    for i in range(l):
        for j in range(i, l):
            if tags_matrix[i][j] != 0:
                pos_list.append([idx2label[tags_matrix[i][j]], i, j])
    return pos_list


def get_spans(tags, idx2label, schema):
    if schema == 'BIO':
        return get_entity_bio(tags, idx2label)
    elif schema == 'span':
        return get_entity_span(tags, idx2label)
    elif schema == 'pointer':
        return get_entity_pointer(tags, idx2label)
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
            try:
                precision = len(self.right_span) / len(self.pred_span)
            except ZeroDivisionError:
                precision = 0.0
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
            try:
                recall = len(self.right_span) / len(self.true_span)
            except ZeroDivisionError:
                recall = 0.0
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
            try:
                recall = len(self.right_span) / len(self.true_span)
            except ZeroDivisionError:
                recall = 0.0
            try:
                precision = len(self.right_span) / len(self.pred_span)
            except ZeroDivisionError:
                precision = 0.0
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
    ## Test BIO
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

    ## Test Span
    label2idx = {'O': 0, 'LOC': 1, 'PER': 2, 'FIN': 3}
    idx2label = {j: i for i, j in label2idx.items()}
    print(get_entity_span([[0, 0, 0, 0], [0, 0, 0, 0]], idx2label))
    print(get_entity_span([[0, 1, 0, 0], [0, 1, 0, 0]], idx2label))  # 单字
    print(get_entity_span([[0, 1, 0, 0], [0, 0, 1, 0]], idx2label))  # 双字
    print(get_entity_span([[0, 1, 0, 0], [0, 0, 1, 1]], idx2label))  # 双字 + 无左边界
    print(get_entity_span([[0, 1, 0, 1], [0, 0, 1, 0]], idx2label))  # 双字 + 无右边界
    print(get_entity_span([[0, 1, 0, 2], [0, 0, 1, 2]], idx2label))  # 双类别
    print(get_entity_span([[0, 0, 1, 0], [0, 1, 0, 0]], idx2label))  # 越界
    print(get_entity_span([[0, 1, 0, 2], [0, 0, 2, 0]], idx2label))  # 不匹配

    ## Test BIO metrics
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
