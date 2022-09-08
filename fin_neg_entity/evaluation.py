# -*-coding:utf-8 -*-

from sklearn.metrics import precision_score, recall_score


def entity_f1(pred_entity, y):
    tp = 0
    fp = 0
    fn = 0
    for pe, y in zip(pred_entity, y):
        if pe != '' and y == 1:
            tp += 1
        elif y == 1 and pe == '':
            fn += 1
        elif y == 0 and pe != '':
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def overall_f1(df):
    df['pred_entity'] = df['text1']
    df.loc[df['pred'] == 0, 'pred_entity'] = ''

    stat = df.groupby('id').agg({'label': sum, 'pred': sum})
    stat['label'] = stat['label'].map(lambda x: int(x > 0))
    stat['pred'] = stat['pred'].map(lambda x: int(x > 0))

    f1_e = entity_f1(df['pred_entity'], df['label'])
    ps = precision_score(stat['label'], stat['pred'])
    rs = recall_score(stat['label'], stat['pred'])
    f1_s = 2 * ps * rs / (ps + rs)

    f1 = 0.6 * f1_e + 0.4 * f1_s
    return {'f1_s': f1_s, 'f1_e': f1_e, 'f1': f1}

