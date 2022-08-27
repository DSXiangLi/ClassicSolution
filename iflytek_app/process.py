# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
from src.evaluation import classification_inference
from src.train_utils import load_checkpoint


def train_process():
    df = pd.read_csv('./trainsample/train.csv')
    df['name'] = df['name'].map(lambda x: x.split(' '))
    df['description'] = df['description'].map(lambda x: x.split(' '))

    df['l1'] = df['name'].map(lambda x: len(x))
    df['l2'] = df['description'].map(lambda x: len(x))
    df['len'] = df.apply(lambda x: x['l1'] + x['l2'], axis=1)

    label = df.groupby('label').size().to_dict()
    label = sorted(label.items(), key=lambda x: x[1])
    label2idx = {j[0]: i for i, j in enumerate(label)}

    print(df.describe())
    print(label2idx)
    return df,  label2idx


def test_process():
    test = pd.read_csv('./trainsample/test.csv')
    test['name'] = test['name'].map(lambda x: x.split(' '))
    test['description'] = test['description'].map(lambda x: x.split(' '))
    test['l1'] = test['name'].map(lambda x: len(x))
    test['l2'] = test['description'].map(lambda x: len(x))
    test['len'] = test.apply(lambda x: x['l1'] + x['l2'], axis=1)

    return test


def result_process(result, label2idx, file_name):
    idx2label = {j:i for i,j in label2idx.items()}
    result = pd.DataFrame(result)
    result['label_id'] = result['pred']
    result['label'] = result['label_id'].map(lambda x: idx2label[x])
    result['id'] = list(range(result.shape[0]))
    result = result.loc[:,['id','label']]
    result.to_csv(file_name, index=False)


def kfold_inference(test_loader, tp, model_cls, ckpt_path, kfold, device):
    prob = {}
    for i in range(kfold):
        model = model_cls(tp)
        ckpt = load_checkpoint(ckpt_path + f'/k{i}')
        model.load_state_dict(ckpt['model_state_dict'])
        result = classification_inference(model, test_loader, device)
        prob[f'prob{i}'] = result['prob']
    result = pd.DataFrame(prob)

    # pred_avg: average prob and argmax
    result['prob'] = result.apply(lambda x: np.average([np.array(x[f'prob{i}']) for i in range(kfold)], axis=0), axis=1)
    result['pred_avg'] = result['prob'].map(lambda x: np.argmax(x))

    # pred_major: major vote on prediction
    for i in range(kfold):
        result[f'pred{i}'] = result[f'prob{i}'].map(lambda x: np.argmax(x, axis=-1))
    result['pred_major'] = result.apply(lambda x: np.argmax(np.bincount([x[f'pred{i}'] for i in range(kfold)])), axis=1)
    return result