# -*-coding:utf-8 -*-

import re
import json
import pandas as pd
from src.preprocess.str_utils import full2half, SpecialToken
from collections import defaultdict
from src.seqlabel_utils import pos2bio
from itertools import chain


def load_event(filename):
    samples = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            trigger_list = []
            if 'event_list' in l:
                for event in l['event_list']:
                    trigger_list.append([event['event_type'], event['trigger']])
                samples.append([l['id'], l['text'], trigger_list])
            else:
                samples.append([l['id'], l['text']])
        if 'event_list' in l:
            return pd.DataFrame(samples, columns=['id', 'text', 'trigger_list'])
        else:
            return pd.DataFrame(samples, columns=['id', 'text'])


def load_argument(filename):
    # split multiple event in 1 sample into multiple samples
    samples = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            trigger_list = []
            if 'event_list' in l:
                for event in l['event_list']:
                    arguments = []
                    for args in event['arguments']:
                        arguments.append([args['role'], args['argument']])
                    samples.append([l['id'], l['text'], event['event_type'], event['trigger'], arguments])
        return pd.DataFrame(samples, columns=['id', 'text', 'event_type', 'trigger', 'arguments'])


class Schema2Label:
    def __init__(self, file_name):
        self.schema = self.load_schema(file_name)
        self._event_bio = None
        self._event = None
        self._event_hier = None
        self._argument_bio = None

    @staticmethod
    def load_schema(filename):
        schema = defaultdict(list)
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                for role in l['role_list']:
                    schema[l['event_type']].append(role['role'])
        return schema

    @property
    def event_label(self):
        if self._event is None:
            self._event = {j: i for i, j in enumerate(self.schema)}
        return self._event

    @property
    def event_bio_label(self):
        if self._event_bio is None:
            bio = {'O': 0}
            for i, j in self.event_label.items():
                bio['B-' + i] = 2 * (j + 1) - 1
                bio['I-' + i] = 2 * (j + 1)
            self._event_bio = bio
        return self._event_bio

    @property
    def argument_bio_label(self):
        if self._argument_bio is None:
            bio = {'O': 0}
            for i, j in enumerate(set(chain(*self.schema.values()))):
                bio['B-' + j] = 2 * (i + 1) - 1
                bio['I-' + j] = 2 * (i + 1)
            self._argument_bio = bio
        return self._argument_bio

    def dump_hierarchy(self):
        # dump hierarchy: {parent: [children]}
        dic = defaultdict(list)
        for l in self.schema:
            dic[l.split('-')[0]].append(l.split('-')[1])
        with open('hierarchy.json', 'w', encoding='utf-') as f:
            f.write(json.dumps(dic, ensure_ascii=False) + '\n')


def gen_pos(text, span_list, special_token=SpecialToken):
    """
    text: 清洗后文本无空格
    span_list: [[span_type, span]]
    """
    pos_list = []
    for span in span_list:
        pattern = span[1]
        for i in special_token:
            if i in pattern:
                pattern = pattern.replace(i, '\\' + i)
        for pos in re.finditer(pattern, text):
            pos = list(pos.span())  # 改成左闭右闭的
            pos_list.append([span[0], pos[0], pos[1] - 1])
    return pos_list


def text_preprocess(s, useless_chars):
    # 注意这里做了大写转小写的转换，如果是比赛需要在最终提交的时候进行还原
    if not s:
        return s
    s = full2half(s)
    # 保证BIO抽取位置对齐
    ## fitler useless chars
    for uc in useless_chars:
        s = s.replace(uc, '')
    ## remove space in text to avoid tokenizer mismatch
    s = re.sub(r'\s{1,}', '', s)
    return s


def event_preprocess(df, useless_chars):
    df['clean_text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))
    if 'trigger_list' in df.columns:
        df['event_pos'] = df.apply(lambda x: gen_pos(x.clean_text, x.trigger_list), axis=1)
        df['event_bio_label'] = df.apply(lambda x: pos2bio(x.clean_text, x.event_pos), axis=1)
        df['event_label'] = df['trigger_list'].map(lambda x: [i[0] for i in x])
    return df


def argument_preprocess(df, useless_chars):
    ##TODO: 修改BIO label 的生成方案
    df['clean_text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))
    df['arguments'] = df['arguments'].map(lambda x: [[i[0], text_preprocess(i[1], useless_chars)]for i in x])
    df['event_text'] = df.apply(lambda x: x.event_type + ':' + x.clean_text, axis=1)
    if 'arguments' in df.columns:
        df['argument_pos'] = df.apply(lambda x: gen_pos(x.event_text, x.arguments), axis=1)
        df['argument_bio_label'] = df.apply(lambda x: pos2bio(x.event_text, x.argument_pos), axis=1)
    return df


def text_alignment(text_o, text_c):
    """
    Input
        text_o: original text
        text_c: text after cleaning
    Return
        pos_map: {org_pos: new_pos }

    """
    pos_map = {}
    i,j=0,0
    lo, lc = len(text_o), len(text_c)
    while i<lo and j < lc:
        if text_o[lo] == text_c[lc]:
            pos_map[lo] = lc
            lo +=1
            lc +=1
        else:
            lo+=1
    return pos_map

