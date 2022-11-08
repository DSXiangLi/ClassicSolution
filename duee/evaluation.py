import numpy as np
import pandas as pd


def extract_multilabel(prob, idx2label, threshold):
    pred = [int(i>threshold) for i in prob]
    labels = []
    for i,j in enumerate(pred):
        if j==1:
            labels.append(idx2label[i])
    return labels


def event_evaluation(y_true, y_pred):
    """
    y_true: [["组织关系-裁员", "组织关系-解散"], ["灾害/意外-车祸", "人生-死亡"], ["竞赛行为-胜负"]]
    y_pred: same as y_true
    """
    tp = 0
    fp = 0
    fn = 0
    n = 0
    for yt, yp in zip(y_true, y_pred):
        yt, yp = set(yt), set(yp)
        tp += len(yt.intersection(yp))
        fp += len(yp.difference(yt))
        fn += len(yt.difference(yp))
        n += len(yt.union(yp))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = tp / n
    return {'n_sample': len(y_true),
            'n_pos': (tp + fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy}


def argument_evaluation(y_true, y_pred):
    """
    Token & Span Level Evaluation
    y_true: 按事件类型explode的样本，每一条是一条文本对应一个事件类型
    """

    pass
