# -*-coding:utf-8 -*-
import os
import json
import pandas as pd
from collections import namedtuple


def single_text(id_list, text_list, label_list, data_dir, output_file):
    Fmt = namedtuple('SingleText', ['id', 'text1', 'label'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w', encoding='utf-8') as f:
        for i, t, l in zip(id_list, text_list, label_list):
            f.write(json.dumps(Fmt(i, t, l)._asdict(), ensure_ascii=False) + '\n')


def double_text(id_list, text_list1, text_list2, label_list, data_dir, output_file):
    """
    双文本输入Iterable
    生成{'text1':'', 'text2':'', 'label':''}
    """
    Fmt = namedtuple('SingleText', ['id', 'text1', 'text2', 'label'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w', encoding='utf-8') as f:
        for i, t1, t2, l in zip(id_list, text_list1, text_list2, label_list):
            f.write(json.dumps(Fmt(i, t1, t2, l)._asdict(), ensure_ascii=False) + '\n')


def single_text_double_label(id_list, text_list, label1_list, label2_list, data_dir, output_file):
    Fmt = namedtuple('SingleText', ['id', 'text1', 'label', 'label2'])

    with open(os.path.join(data_dir, output_file + '.txt'), 'w', encoding='urf-8') as f:
        for i, t, l1, l2 in zip(id_list, text_list, label1_list, label2_list):
            f.write(json.dumps(Fmt(i, t, l1, l2)._asdict(), ensure_ascii=False) + '\n')


def json2df(file):
    lines = []
    with open(file, 'r') as f:
        for i in f.readlines():
            lines.append(json.loads(i))
    df = pd.DataFrame(lines)
    return df
