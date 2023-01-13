# -*-coding:utf-8 -*-
import torch
import numpy as np
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, Recall, Precision
from src.seqlabel_utils import SpanF1, SpanPrecision, SpanRecall, PointerF1, PointerPrecision, PointerRecall
from itertools import chain


def binary_cls_metrics(model, valid_loader, device, threshold=0.5, label_name='label'):
    """
    Binary Classification Metircs， support
    - Macro/micro average
    - F1, precision, recall
    - average precision
    - AUC
    - Accuracy
    - loss
    """
    model.eval()

    metrics = {
        'acc': Accuracy(threshold=threshold, num_classes=2).to(device),
        'auc': AUROC(num_classes=2).to(device),
        'ap': AveragePrecision(num_classes=2).to(device),
        'f1': F1Score(num_classes=2).to(device),
        'recall': Recall(threshold=threshold, num_classes=2).to(device),
        'precision': Precision(threshold=threshold, num_classes=2).to(device)
    }
    val_loss = []

    for batch in valid_loader:
        features = {k:v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            logits = model(features)
            loss = model.compute_loss(features, logits)
        val_loss.append(loss.item())

        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = (probs[:, 1] > threshold).int().to(device)
        for metric in metrics.values():
            if metric in ['auc', 'ap']:
                metric.update(probs, features[label_name])
            else:
                metric.update(preds, features[label_name])

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def multi_cls_metrics(model, valid_loader, device, label_name='label'):
    """
    Multi class classification Metics
    - Macro/micro average
    - F1, precision, recall
    - average precision: micro only
    - AUC: micro only
    - Accuracy
    - loss
    """
    model.eval()

    # Tracking variables
    metrics = {}
    for avg in ['micro', 'macro']:
        metrics.update({
            f'acc_{avg}': Accuracy(average=avg, num_classes=model.label_size).to(device),
            f'f1_{avg}': F1Score(average=avg, num_classes=model.label_size).to(device),
            f'recall_{avg}': Recall(average=avg, num_classes=model.label_size).to(device),
            f'precision_{avg}': Precision(average=avg, num_classes=model.label_size).to(device)
        })
    # for AUC and AP only macro average is supported for multiclass
    metrics.update({
        f'auc_macro': AUROC(average='macro', num_classes=model.label_size).to(device),
        f'ap_macro': AveragePrecision(average='macro', num_classes=model.label_size).to(device),

    })
    val_loss = []

    for batch in valid_loader:
        features = {k:v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            logits = model(features)
            loss = model.compute_loss(features, logits)

        val_loss.append(loss.item())

        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        for metric in metrics.values():
            if 'auc' in metric or 'ap' in metric:
                metric.update(probs, features[label_name])
            else:
                metric.update(preds, features[label_name])

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def multilabel_metrics(model, valid_loader, device, label_name='label'):
    """
    Global Average Multi Label Classification
    """
    model.eval()

    # Tracking variables
    metrics = {}
    metrics.update({
        f'acc': Accuracy( num_classes=model.label_size, mdmc_average='global').to(device),
        f'f1': F1Score( num_classes=model.label_size, mdmc_average='global').to(device),
        f'recall': Recall(num_classes=model.label_size, mdmc_average='global').to(device),
        f'precision': Precision(num_classes=model.label_size, mdmc_average='global').to(device),
    })
    val_loss = []

    for batch in valid_loader:
        features = {k:v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            logits = model(features)
            loss = model.compute_loss(features, logits)

        val_loss.append(loss.item())

        for metric in metrics.values():
            metric.update(logits, features[label_name].int())

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def seq_tag_metrics(model, valid_loader, idx2label, schema, device):
    """
    Sequence Labelling task seq level metircs, supported
    - Macro/micro average
    - F1, precision, recall
    - Accuracy
    - loss
    """
    model.eval()

    # Tracking variables
    metrics = {}
    for avg in ['micro', 'macro']:
        metrics.update({
            f'f1_{avg}': SpanF1(idx2label=idx2label, schema=schema, avg=avg),
            f'recall_{avg}': SpanRecall(idx2label=idx2label, schema=schema, avg=avg),
            f'precision_{avg}': SpanPrecision(idx2label=idx2label, schema=schema, avg=avg)
        })
    val_loss = []

    for batch in valid_loader:
        features = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(features)
            loss = model.compute_loss(features, logits)
            preds = model.decode(features, logits)

        val_loss.append(loss.item())
        mask = features['attention_mask'].view(-1) == 1
        label_ids = features['label_ids'].view(-1)[mask].cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.view(-1)[mask].cpu().numpy()
        else:
            # CRF decoder return List[List] with real seq len
            preds = list(chain(*preds))
        for metric in metrics.values():
            metric.update(preds, label_ids)

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def seq_span_metrics(model, valid_loader, idx2label, device):
    """
    Sequence Labelling task seq level metircs, supported
    - Macro/micro average
    - F1, precision, recall
    - Accuracy
    - loss
    """
    model.eval()

    # Tracking variables
    metrics = {}
    for avg in ['micro', 'macro']:
        metrics.update({
            f'f1_{avg}': SpanF1(idx2label=idx2label, schema='span', avg=avg),
            f'recall_{avg}': SpanRecall(idx2label=idx2label, schema='span', avg=avg),
            f'precision_{avg}': SpanPrecision(idx2label=idx2label, schema='span', avg=avg)
        })
    val_loss = []

    for batch in valid_loader:
        features = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(features)
            loss = model.compute_loss(features, logits)
            start_pred, end_pred = model.decode(features, logits)

        val_loss.append(loss.item())
        mask = features['attention_mask'].view(-1) == 1
        start_label = features['label_start'].view(-1)[mask].cpu().numpy()
        end_label = features['label_end'].view(-1)[mask].cpu().numpy()
        start_pred = start_pred.view(-1)[mask].cpu().numpy()
        end_pred = end_pred.view(-1)[mask].cpu().numpy()
        for metric in metrics.values():
            metric.update((start_pred, end_pred), (start_label, end_label))

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics



def seq_pointer_metrics(model, valid_loader, idx2label, device):
    """
    Global Pointer 返回预测是二分类，可以直接复用binary classification
    """
    model.eval()
    # Tracking variables
    metrics = {}
    for avg in ['micro', 'macro']:
        metrics.update({
            f'f1_{avg}': PointerF1(head_size=len(idx2label), device=device, avg=avg),
            f'recall_{avg}': PointerRecall(head_size=len(idx2label), device=device, avg=avg),
            f'precision_{avg}': PointerPrecision(head_size=len(idx2label), device=device, avg=avg)
        })
    val_loss = []

    for batch in valid_loader:
        features = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(features)
            loss = model.compute_loss(features, logits)
            pred = model.decode(features, logits)
            mask = model.get_mask(features, logits)

        val_loss.append(loss.item())
        label_ids = features['label_ids']
        preds = pred * mask.long()  # mask all padding and lower triangle to zero

        for metric in metrics.values():
            metric.update(preds, label_ids)

    multi_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    multi_metrics['val_loss'] = np.mean(val_loss)
    return multi_metrics


def tag_cls_log(epoch, tag_metrics):
    print("\n")
    print(f"{'Epoch':^7} | {'Macro Precision':^15} | {'Macro Recall':^12} | {'Macro F1':^9}")
    print('-' * 70)
    print(f"{epoch + 1:^7}  | {tag_metrics['precision_macro']:^15.3%} |",
          f"{tag_metrics['recall_macro']:^12.3%} | {tag_metrics['f1_macro']:^9.3%} ")

    print(f"{'Epoch':^7}  | {'Micro Precision':^15} | {'Micro Recall':^12} | {'Micro F1':^9}")
    print('-' * 70)
    print(f"{epoch + 1:^7} | {tag_metrics['precision_micro']:^15.3%} |",
          f"{tag_metrics['recall_micro']:^12.3%} | {tag_metrics['f1_micro']:^9.3%} ")
    print("\n")


def binary_cls_log(epoch, binary_metrics):
    print("\n")
    print(f"{'Epoch':^7} | {'Val Acc':^9} | {'Val AUC':^9} | {'Val AP':^9} | {'Precision':^9} | {'Recall':^9} | {'Val F1':^9}")
    print('-'*80)
    print(f"{epoch + 1:^7} | {binary_metrics['acc']:^9.3%} | {binary_metrics['auc']:^9.3%} |",
          f"{binary_metrics['ap']:^9.3%} | {binary_metrics['precision']:^9.3%} |",
          f"{binary_metrics['recall']:^9.3%} | {binary_metrics['f1']:^9.3%} ")
    print("\n")


def multilabel_log(epoch, multilabel_metrics):
    print("\n")
    print(f"{'Epoch':^7} | {'Val Acc':^9} | {'Precision':^9} | {'Recall':^9} | {'Val F1':^9}")
    print('-' * 80)
    print(f"{epoch + 1:^7} |{multilabel_metrics['acc']:^9.3%} | {multilabel_metrics['precision']:^9.3%}  |",
          f"{multilabel_metrics['recall']:^9.3%} | {multilabel_metrics['f1']:^9.3%} ")
    print("\n")


def multi_cls_log(epoch, multi_metrics):
    print("\n")
    print(f"{'Epoch':^7} | {'Macro Acc':^9} | {'Macro AUC':^9} | {'Macro AP':^9} |",
          f"{'Macro Precision':^15} | {'Macro Recall':^12} | {'Macro F1':^9}")
    print('-' * 90)
    print(f"{epoch + 1:^7} | {multi_metrics['acc_macro']:^9.3%} | {multi_metrics['auc_macro']:^9.3%} |",
          f"{multi_metrics['ap_macro']:^9.3%} | {multi_metrics['precision_macro']:^15.3%} |",
          f"{multi_metrics['recall_macro']:^12.3%} | {multi_metrics['f1_macro']:^9.3%} ")

    print(f"{'Epoch':^7} | {'Micro Acc':^9} | {'Micro AUC':^9} | {'Micro AP':^9} |",
          f"{'Micro Precision':^15} | {'Micro Recall':^12} | {'Micro F1':^9}")
    print('-' * 90)
    print(f"{epoch + 1:^7} | {multi_metrics['acc_micro']:^9.3%} | {'-':^9} |",
          f"{'-':^9} | {multi_metrics['precision_micro']:^15.3%} |",
          f"{multi_metrics['recall_micro']:^12.3%} | {multi_metrics['f1_micro']:^9.3%} ")
    print("\n")
