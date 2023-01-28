# -*-coding:utf-8 -*-
"""
    Export Test file for Duee Evaluation
"""
import json
import pandas as pd
import os
import ast
from src.seqlabel_utils import extract_entity
from duee.mrc_query import gen_event
from pathlib import Path
from zipfile import ZipFile

DIR = os.path.join(Path(__file__).absolute().parent, 'trainsample')


def format_argument(pred_arg, event_type):
    output = {'event_type': event_type, 'arguments': []}
    for key, val in pred_arg.items():
        for i in val:
            output['arguments'].append({'role': key.split('-')[-1], 'argument': i})
    return output


def merge_dict(dict_list):
    dic = dict_list[0]
    for i in dict_list:
        dic.update(i)
    return dic


def process(pred_file, is_query, zip_file):
    test = pd.read_csv(os.path.join(DIR, pred_file))
    test['pred_pos'] = test['pred_pos'].map(lambda x: ast.literal_eval(x))
    test['event_type'] = test['text1'].map(lambda x: x.split(':')[0])
    test['pred_arg'] = test.apply(lambda x: extract_entity(x.text1, x.pred_pos), axis=1)
    if is_query:
        # MRC Format
        test['event_type'] = test['event_type'].map(lambda x: gen_event(x))
        test = test.loc[test['pred_pos'].map(lambda x: len(x)>0),:] # 过滤无抽取argument
        test = test.groupby(['event_type', 'id']).agg({'pred_arg': list})
        test['pred_arg'] = test['pred_arg'].map(lambda x: merge_dict(x))
        test.reset_index(inplace=True)
    test['arguments'] = test.apply(lambda x: format_argument(x.pred_arg, x.event_type), axis=1)
    test = test.groupby('id').agg({'arguments': list})
    test.reset_index(inplace=True)

    with open(os.path.join(DIR, 'duee.json'), 'w', encoding='utf-8') as f:
        for idx, event in zip(test['id'], test['arguments']):
            f.write(json.dumps({'id': idx, 'event_list': event}, ensure_ascii=False) + '\n')

    with ZipFile(os.path.join(DIR, zip_file), 'w') as z:
       z.write(os.path.join(DIR, 'duee.json'), arcname='duee.json')  # zipping the file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--is_query', default=False, action='store_true')
    parser.add_argument('--zip_file', type=str)

    args = parser.parse_args()
    process(args.input_file, args.is_query, args.zip_file)