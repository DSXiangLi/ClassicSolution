# -*-coding:utf-8 -*-

import re
import json
import pandas as pd
from src.preprocess.str_utils import full2half, SpecialToken
from collections import defaultdict
from itertools import chain
from duee.mrc_query import gen_query


def load_event(filename):
    samples = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            trigger_list = []
            if 'event_list' in l:
                for event in l['event_list']:
                    trigger_list.append([event['event_type'], event['trigger'], event['trigger_start_index']])
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
            if 'event_list' in l:
                for event in l['event_list']:
                    arguments = []
                    for args in event['arguments']:
                        arguments.append([args['role'], args['argument'], args['argument_start_index']])
                    samples.append([l['id'], l['text'], event['event_type'], event['trigger'], arguments])
        return pd.DataFrame(samples, columns=['id', 'text', 'event_type', 'trigger', 'arguments'])


class Schema2Label:
    def __init__(self, file_name):
        self.schema = self.load_schema(file_name)
        self._event_bio = None
        self._event = None
        self._argument_bio = None
        self._event_argument = None
        self._event_argument_bio = None

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

    @property
    def event_argument_bio_label(self):
        if self._event_argument_bio is None:
            bio = {'O':0}
            i=0
            for event, args in self.schema.items():
                for argument in args:
                    bio['-'.join(['B', event, argument])] = 2* (i+1)-1
                    bio['-'.join(['I', event, argument])] = 2* (i+1)
                    i+=1
            self._event_argument_bio = bio
        return self._event_argument_bio

    @property
    def event_argument_label(self):
        if self._event_argument is None:
            idx = 0
            label = {}
            for event, arguments in self.schema.items():
                for argument in arguments:
                    label[event + '-' + argument] = idx
                    idx += 1
            self._event_argument = label
        return self._event_argument


def text_preprocess(s, useless_chars):
    # 注意这里做了大写转小写的转换，如果是比赛需要在最终提交的时候进行还原
    if not s:
        return s
    s = str(s)
    s = full2half(s)
    # 保证BIO抽取位置对齐
    ## fitler useless chars
    for uc in useless_chars:
        s = s.replace(uc, '')
    ## remove space in text to avoid tokenizer mismatch
    s = re.sub(r'\s{1,}', '', s)
    return s


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


def text_alignment(text_o, text_c):
    """
    Input
        text_o: original text
        text_c: text after cleaning
    Return
        pos_map: {org_pos: new_pos }

    """
    pos_map = {}
    i, j = 0, 0
    lo, lc = len(text_o), len(text_c)
    while i < lo and j < lc:
        if text_o[i] == text_c[j]:
            pos_map[i] = j
            i += 1
            j += 1
        elif text_o[i] == ' ':
            # 原始文本存在空格需要跳过
            pos_map[i] = j
            i += 1
        else:
            i += 1
    return pos_map


def adjust_pos(pos_map, span_list):
    """
    基于清洗后文本，返回事件触发词列表
    Input
        pos_map: 清洗后文本和原始文本位置对应关系
        span_list: [span_type, span, start_index]
    Return:
        pos_list: [span_type, start_index, end_index] 左闭右闭
    """
    pos_list = []
    for s in span_list:
        span = re.sub(r'\s{1,}', '', s[1])  # remove all spances in span
        start = pos_map[s[2]]
        end = start + len(span) - 1
        pos_list.append([s[0], start, end])
    return pos_list


def check(text, pos):
    tmp = []
    for i in pos:
        tmp.append(text[i[1]:i[2] + 1])
    return tmp


def event_preprocess(df, useless_chars):
    df['clean_text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))

    if 'trigger_list' in df.columns:
        # 使用正则直接搜索Trigger词，会有5%左右是词相同但是词出现的上下文并非是事件的情况发生
        # df['event_pos'] = df.apply(lambda x: gen_pos(x.clean_text, x.trigger_list), axis=1)
        df['pos_map'] = df.apply(lambda x: text_alignment(x.text, x.clean_text), axis=1)
        df['event_pos'] = df.apply(lambda x: adjust_pos(x.pos_map, x.trigger_list), axis=1)

        # 检查是否存在调整后的位置抽取实体错误的情况
        df['check'] = df.apply(lambda x: check(x.clean_text, x.event_pos), axis=1)
        counter = sum(
            df['trigger_list'].map(lambda x: [re.sub(r'\s{1,}', '', i[1]) for i in x]) != df['check'])
        print(f'{counter} out of {df.shape[0]} trigger not match')

        # 检查是否存在label越界的情况
        counter = df.apply(lambda x: max([i[2] for i in x.event_pos]) >= len(x.clean_text), axis=1).sum()
        print(f'{counter} out of {df.shape[0]} even pos exceed text line')

        # 生成事件分类多标签
        df['event_label'] = df['trigger_list'].map(lambda x: list(set([i[0] for i in x])))
    return df


def argument_preprocess(df, useless_chars):
    """
    预测事件+原始文本 -> 预测事件对应Argument
    Pipeline抽取：已知event，对应argument的BIO标签
        - df: 一条样本是event + text唯一确定
    """
    df['text'] = df['text'].map(lambda x: full2half(x))  # for following al
    df['clean_text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))

    # 文本清洗会影响argument位置，生成位置映射
    df['pos_map'] = df.apply(lambda x: text_alignment(x.text, x.clean_text), axis=1)

    # 拼接事件类型
    df['event_text'] = df.apply(lambda x: x.event_type + ':' + x.clean_text, axis=1)
    df['pos_map'] = df.apply(lambda x: {i: j + len(x.event_type) + 1 for i, j in x.pos_map.items()}, axis=1)
    # 生成位置调整后的BIO起始位置
    df['arguments'] = df.apply(lambda x: [[x.event_type + '-' + i[0], text_preprocess(i[1], useless_chars), i[2]]\
                                          for i in x.arguments], axis=1)
    df['arguments_pos'] = df.apply(lambda x: adjust_pos(x.pos_map, x.arguments), axis=1)
    return df


def argument_mrc_preprocess(df, useless_chars, event2argument, filter_neg=False):
    """
    预测事件Argument Query + 原始文本 -> 预测事件对应Argument
    Pipeline抽取：已知event，对应argument的BIO标签
        - df: 一条样本是event + text唯一确定
    """
    df['text'] = df['text'].map(lambda x: full2half(x))  # for following al
    df['clean_text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))
    # 文本清洗会影响argument位置，生成位置映射
    df['pos_map'] = df.apply(lambda x: text_alignment(x.text, x.clean_text), axis=1)

    # 拆分事件对应论元成单条样本，并过滤该论元对应的label
    df['argument_type'] = df['event_type'].map(lambda x: event2argument[x])
    df = df.explode('argument_type')

    df['query'] = df.apply(lambda x: gen_query(x.event_type, x.argument_type), axis=1)
    df['text'] = df.apply(lambda x: x.query + ':' + x.clean_text, axis=1)

    # 生成位置调整后的BIO起始位置
    if 'arguments' in df.columns:
        # 过滤有论元样本
        df['arguments'] = df.apply(lambda x: [i for i in x.arguments if i[0] == x.argument_type], axis=1)
        df['arguments'] = df.apply(lambda x: [[x.event_type + '-' + i[0], text_preprocess(i[1], useless_chars), i[2]]\
                                              for i in x.arguments], axis=1)
        df['pos_map'] = df.apply(lambda x: {i: j + len(x.query) + 1 for i, j in x.pos_map.items()}, axis=1)
        df['arguments_pos'] = df.apply(lambda x: adjust_pos(x.pos_map, x.arguments), axis=1)
        if filter_neg:
            df = df.loc[df['arguments_pos'].map(lambda x: len(x) > 0), :]
    return df


def joint_preprocess(df, useless_chars):
    """
    事件和论元联合抽取
        - df: 一条样本是event + text唯一确定
        - event：事件多标签
        - argument：event+argument的论元span标签
    """
    df['text'] = df['text'].map(lambda x: full2half(x))  # for following al
    df['clean_text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))
    if 'arguments' in df.columns:
        df['pos_map'] = df.apply(lambda x: text_alignment(x.text, x.clean_text), axis=1)
        df['arguments'] = df['arguments'].map(lambda x: [[i[0], text_preprocess(i[1], useless_chars), i[2]] for i in x])
        df['arguments_pos'] = df.apply(lambda x: adjust_pos(x.pos_map, x.arguments), axis=1)
        df['arguments_pos'] = df.apply(lambda x: [[x.event_type + '-' + i[0], i[1], i[2]] for i in x.arguments_pos],
                                       axis=1)
        # 把展开的event+text重新聚合到text粒度
        df = df.groupby(['id', 'text', 'clean_text']).agg({'event_type': list,
                                                           'arguments_pos': lambda x: list(chain(*x))})
        df.reset_index(inplace=True)
    return df
