# -*-coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
from src.preprocess.str_utils import full2half, mention_handler, get_useless_chars


def fix_entity(entity):
    # 实体清洗
    if not entity:
        return entity
    # 剔除2个以上的？
    entity = re.sub('\?{2,}', '', entity)
    # 消重
    entity = ';'.join(sorted(set(entity.split(';'))))
    # full2half
    entity = full2half(entity)
    return entity


def find_ent_pos(text, entities):
    # 定位文本中所有出现实体的位置返回（begin_pos, end_pos) list
    pos_list = []
    if not entities:
        return pos_list

    for ent in entities.split(';'):
        try:
            ent = ent.replace('(', '\(').replace(')', '\)')
            for pos in re.finditer(ent, text):
                pos_list.append(pos.span())
        except Exception as e:
            print(e, entities)
    return pos_list


def false_ent(text, entities):
    # 定位实体全部识别错误的样本并剔除
    if not entities:
        return False
    flag = any((ent in text for ent in entities.split(';')))
    return not flag


def text_preprocess(s, useless_chars):
    if not s:
        return s

    s = full2half(s)
    # 图片
    s = re.sub('\{IMG:.?.?.?\}', ' ', s)
    # http
    s = re.sub('(https?|ftp|file|www)(:\/{2}|\.)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', s)  # 过滤网址
    s = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', s).replace("()", "")  # 过滤www开头的m
    # http tag
    s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签

    s = re.sub('^\?+', '', s)  # 问号开头直接删除
    s = re.sub('\?{2,}', ' ', s)  # 中间出现多余1个问号用空格替代

    s = re.sub('^#', '', s)  # 井号开头直接删除
    s = re.sub('#+', ' ', s)  # 井号在中间用空格分隔

    s = re.sub(re.compile('微信[:：]?[a-zA-Z0-9]+'), ' ', s)  # 微信
    s = re.sub(re.compile('(\d{4}-\d{2}-\d{2})|(\d{2}:\d{2}:\d{2})'), ' ', s)  # 时间

    s = re.sub(mention_handler.re_pattern, ' ', s)  # @

    # fitler useless chars
    for uc in useless_chars:
        s = s.replace(uc, '')
    return s


def hierarchy_text_split(text):
    sentences = []
    tmp = []
    # 优先使用句子分割
    for s in text:
        tmp.append(s)
        if s in {'。', '！', '!', '？', '?'}:
            sentences.append(''.join(tmp))
            tmp = []
    # 如果没有则使用逗号分隔
    if len(sentences) <= 1:
        for s in text:
            tmp.append(s)
            if s in {',', '，'}:
                sentences.append(''.join(tmp))
                tmp = []
    if len(tmp) > 0:
        sentences.append(''.join(tmp))
    return sentences


def split_text(title, text, max_seq_len):
    """
    text按标点符号超过长度后进行split，
    """
    ## 满足长度的直接返回
    if not title or title in text:
        seq_len = max_seq_len
        title = ''
        if len(text) < seq_len:
            return [text]
    else:
        seq_len = max_seq_len - len(title)
        if len(text) < seq_len:
            return [title + ' ' + text]
    ## 不满足长度的按句子进行拆分后再和title拼接得到多段文本返回
    sentences = hierarchy_text_split(text)
    corpus = []
    tmp = title
    for i in range(len(sentences)):
        if len(tmp) + len(sentences[i]) > seq_len:
            corpus.append(tmp)
            tmp = title
            i = max(1, i - 2)  # 回退两个句子，虽然corpus间会有部分文本重复，但是会保留切分的各个句子更全面的上下文
        tmp += sentences[i]
    if len(tmp) > len(title):
        corpus.append(tmp)
    return corpus


def data_process(file_name='./trainsample/Train_Data.csv'):
    df = pd.read_csv(file_name)
    df.fillna({'text': '', 'title': '', 'unknownEntities': ''}, inplace=True)

    # 定位无用符号
    useless_chars = get_useless_chars(list(df['text'].values) + list(df['title'].values))
    # 文本清洗
    df['text'] = df['text'].map(lambda x: text_preprocess(x, useless_chars))
    df['title'] = df['title'].map(lambda x: text_preprocess(x, useless_chars))
    df['entities'] = df['unknownEntities'].map(lambda x: fix_entity(x))
    df['false_ent'] = df.apply(lambda x: false_ent(x.title + x.text, x.entities), axis=1)
    df['text_l'] = df['text'].map(lambda x: len(x))
    df['title_l'] = df['title'].map(lambda x: len(x))
    print(df.describe())
    print(f"{df['entities'].map(lambda x: len(x) == 0).sum()}样本无实体")
    print(f"{df['false_ent'].sum()}样本定位的实体未出现在文本中为错误样本")
    print(f"{sum(df['text_l'] > 510)}样本文本长度超过510")
    print(f"{sum(df['title_l'] > 510)}样本标题长度超过510")

    # 过滤错误样本
    df = df.loc[~df['false_ent'], :]
    # 文本split
    df['corpus'] = df.apply(lambda x: split_text(x.title, x.text, max_seq_len=510), axis=1)
    df = df.explode('corpus').reset_index(drop=True)
    df['ent_pos'] = df.apply(lambda x: find_ent_pos(x.corpus, x.entities), axis=1)
    # 过滤split之后无实体的样本
    df = df.loc[~(df['entities'].map(lambda x: len(x)>0) & df['ent_pos'].map(lambda x:len(x)==0)),:]
    print(f'切分样本后总共样本数{df.shape[0]}')
    return df


def bio_tag(text, entity_pos_list):
    label = ['O'] * len(text)
    for pos in entity_pos_list:
        label[pos[0]] = 'B'
        label[(pos[0]+1):pos[1]] = ['I'] * (pos[1]-pos[0]-1)
    return label