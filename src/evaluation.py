# -*-coding:utf-8 -*-
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score,\
classification_report
from seqeval.metrics import classification_report as span_cls_report
from itertools import chain


def classification_inference(model, data_loader, device):
    model.eval()

    all_preds = []
    all_probs = []
    for batch in data_loader:
        # Load batch to GPU
        inputs = {k: v.to(device) for k,v in batch.items()}

        # Compute logits
        with torch.no_grad():
            logits = model(inputs) # ignore label for test
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        all_preds += np.argmax(probs, axis=-1).tolist()
        all_probs += probs.tolist()

    output = {
        'pred': all_preds,
        'prob': all_probs
    }
    return output


def seqlabel_inference(model, data_loader, device):
    """
    Sequence Labeling Inference
        Return: preds list[list(real seq len)]
    """
    model.eval()

    all_preds = []
    # for seqlbel predict is done on
    for batch in data_loader:
        # Load batch to GPU
        features = {k:v.to(device) for k,v in batch.items()}
        # Compute logits
        with torch.no_grad():
            logits = model(features)
            preds = model.decode(features, logits)
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        preds = [pred[1:] for pred in preds] # remove CLS
        all_preds.extend(preds)
    return all_preds


def binary_cls_report(probs, labels, thresholds):
    """
    二分类任务Evaluation
        probs: (n_samples, 2)
        labels: (n_samples,)
        threhoslds: 计算不同阈值下的precision，recall和f1
    """
    probs = [i[1] for i in probs]
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    n_sample = len(probs)
    n_pos = sum(labels)
    # Precision & Recall by threshold
    result = []
    for thr in thresholds:
        tmp = [int(i > thr) for i in probs]
        precision = precision_score(labels, tmp)
        recall = recall_score(labels, tmp)
        accuracy = accuracy_score(labels, tmp)
        result.append((thr, sum(tmp), precision, recall, accuracy, auc, ap, n_sample, n_pos))

    df = pd.DataFrame(result, columns=['threshold', 'n', 'precision', 'recall', 'accuracy', 'auc', 'ap','total','total_pos'])
    df = df.to_string(formatters={'threhsold': "{:.2f}".format,
                                  'n': "{0:d}".format, 'precision': "{:.1%}".format,
                                  'recall': "{:.1%}".format, 'accuracy': "{:.1%}".format,
                                  'auc': "{:.1%}".format,'ap': "{:.1%}".format,
                                  'total': '{0:d}'.format, 'total_pos': '{0:d}'.format
                                  })
    return df


def multi_cls_report(probs, labels, idx2label):
    """
    多分类任务 Evaluation
        probs: (n_samples, label_size)
        labels: (n_samples,)
        idx2label: labelid 到分类名称的映射
    支持
    1. Overall Accuracy, AUC, AP
    2. 分label的precision， recall，f1
    3. micro, macro: precision, recall, f1
    """
    predictions = np.argmax(probs, axis=-1)
    label_names = idx2label.values()
    report = classification_report(labels, predictions, target_names=label_names)
    return report


def seqlabel_report(preds, labels, idx2label):
    """
    Sequence Label task evalutaion 
    preds: list[list(real seq len)] sequence label prediction 
    labels: list[list(real seq len)]
    idx2label: label_id. to label_name mapping
    """
    preds = list(chain(*preds))
    labels = list(chain(*labels))
    
    tag_report = classification_report(labels, preds, target_names=idx2label.values())
    labels = [idx2label[i] for i in labels]
    preds = [idx2label[i] for i in preds]
    span_report = span_cls_report([labels], [preds], digits=3)
    
    text = '='*20 + 'Tag level report' + '='*20 + '\n\n'
    text += tag_report + '\n\n'
    text += '='*20 + 'Span level report' + '='*20 + '\n\n'
    text += span_report
    return text
