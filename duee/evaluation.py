import torch.nn.functional as F
import torch
import numpy as np


def extract_multilabel(prob, idx2label, threshold, greedy=False):
    pred = [int(i>threshold) for i in prob]
    if sum(pred)==0 and greedy:
        return [idx2label[ np.argmax(prob)]]
    labels = []
    for i,j in enumerate(pred):
        if j==1:
            labels.append(idx2label[i])
    return labels


def multilabel_evaluation(y_true, y_pred):
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


def multilabel_inference(model, data_loader, device):
    model.eval()

    all_probs = []
    for batch in data_loader:
        # Load batch to GPU
        inputs = {k: v.to(device) for k, v in batch.items()}

        # Compute logits
        with torch.no_grad():
            logits = model(inputs)  # ignore label for test
            probs = F.sigmoid(logits).cpu().numpy()
        all_probs += probs.tolist()
    return all_probs