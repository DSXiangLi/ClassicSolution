# -*-coding:utf-8 -*-
from collections import defaultdict
import pandas as pd
import re

from src.seqlabel_utils import get_entity_bio, extract_entity

def ispair(x):
    counter = 0
    for i in x:
        if '(' in x:
            counter += 1
        elif ')' in x:
            if counter == 0:
                return False
            else:
                counter -= 1
    if counter != 0:
        return False
    else:
        return True


def islegal(x):
    invalid_chars = {' ', '&', ',', '?', '[', ']', '·'}
    if len(x) <= 1:
        return False
    elif re.search('^[\-|\.|\·|\&|\+|\:|\?]', x):
        return False
    elif re.search('$[\-|\.|\·|\&|\+|\:|\?]', x):
        return False
    elif re.findall('\\' + "|\\".join(invalid_chars), x):
        return False
    elif re.findall('[\(|\)]', x):
        if ispair(x):
            return True
        else:
            return False
    else:
        return True


def overall_f1(true_set_list, pred_set_list):
    tp, fp, fn = 0, 0, 0
    for t, p in zip(true_set_list, pred_set_list):
        tp += len(t.intersection(p))
        fp += len(p.difference(t))
        fn += len(t.difference(p))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    stat = pd.DataFrame({'precision': precision, 'recall': recall, 'f1': f1}, index=[0])
    return stat


def aggregate_f1(ids, true_entity, pred_entity, known_entity):
    true = defaultdict(set)
    pred = defaultdict(set)
    for id, t, p in zip(ids, true_entity, pred_entity):
        for i in t.split(';'):
            true[id].add(i)
        for i in p:
            pred[id].add(i)
    df = pd.DataFrame({'id': list(set(ids))})
    df['true'] = df['id'].map(lambda x: true[x])
    df['pred'] = df['id'].map(lambda x: pred[x])
    df['true_new'] = df['true'].map(lambda x: set([i for i in x if i not in known_entity]))
    df['pred_new'] = df['pred'].map(lambda x: set([i for i in x if i not in known_entity]))
    print('All Entity Evaluation')
    print(overall_f1(df['true'].values, df['pred'].values))
    print('Unknown Entity Evalutation')
    print(overall_f1(df['true_new'].values, df['pred_new'].values))
    return df
