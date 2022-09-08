# -*-coding:utf-8 -*-
import os
import json
import pandas as pd
from collections import namedtuple


def single_text(text_list, label_list, data_dir, output_file):
    Fmt = namedtuple('SingleText', ['text1', 'label'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w') as f:
        for t, l in zip(text_list, label_list):
            f.write(json.dumps(Fmt(t, l)._asdict(), ensure_ascii=False) + '\n')


def double_text(text_list1, text_list2, label_list, data_dir, output_file):
    """
    双文本输入Iterable
    生成{'text1':'', 'text2':'', 'label':''}
    """
    Fmt = namedtuple('SingleText', ['text1', 'text2', 'label'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w') as f:
        for t1, t2, l in zip(text_list1, text_list2, label_list):
            f.write(json.dumps(Fmt(t1, t2, l)._asdict(), ensure_ascii=False) + '\n')


def single_text_double_label(text_list, label1_list, label2_list, data_dir, output_file):
    Fmt = namedtuple('SingleText', ['text1', 'label','label2'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w') as f:
        for t, l1, l2 in zip(text_list, label1_list, label2_list):
            f.write(json.dumps(Fmt(t, l1, l2)._asdict(), ensure_ascii=False) + '\n')


def json2df(file):
    lines = []
    with open(file, 'r') as f:
        for i in f.readlines():
            lines.append(json.loads(i))
    df = pd.DataFrame(lines)
    return df
