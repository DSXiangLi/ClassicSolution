# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
import re
from src.preprocess.str_utils import full2half
from sklearn.model_selection import train_test_split

def filter_entity(entity, text):
    # 过滤重复，单字，空实体, 不在文本中的错误实体
    entity = entity.split(';')
    output = set()
    for i in entity:
        if i.strip() and len(i.strip()) > 1 and i in text:
            output.add(i)

    # 过滤嵌套实体只保留最长实体span
    entity = sorted(output, key=lambda x: len(x))
    new_entity = entity.copy()
    for i in range(len(entity)):
        e1 = entity[i]
        for e2 in entity[(i + 1):]:
            if e1 in e2:
                # 如果e1是独立出现则保留，只过滤嵌套的子实体
                if e1 not in text.replace(e2, ''):
                    new_entity.remove(e1)
                    break
    return new_entity


def has_all_entity(text, entity_list):
    return all((i in text for i in entity_list))


def text_preprocess(s):
    if not s:
        return s

    s = full2half(s)
    s = re.sub('\{IMG:.?.?.?\}', ' ', s)  # 图片

    s = re.sub(re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'), ' ', s)  # 网址

    s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签

    s = re.sub(r'&[a-zA-Z0-9]{1,4}', ' ', s)  # &nbsp  &gt  &type &rdqu   ....

    s = re.sub('^\?+', '', s)  # 问号开头直接删除
    s = re.sub('\?{2,}', ' ', s)  # 中间出现多余1个问号用空格替代

    s = re.sub('^#', '', s)  # 井号开头直接删除
    s = re.sub('#+', ' ', s)  # 井号在中间用空格分隔

    s = re.sub('\[超话\]', ' ', s)  # 超话删除
    return s


def find_pos(text, entity):
    if not text:
        return -1
    return text.find(entity)


def merge_text1(title, text):
    # 用于实体过滤的标题和文本拼接逻辑
    if not title:
        return text
    if not text:
        return title
    return title + ' ' + text


def merge_text2(title, text, entity, title_pos, text_pos):
    max_seq_len = 512 - 3 - len(entity)
    s = ''
    if not title:
        if text_pos > max_seq_len:
            s += text[text_pos - int(max_seq_len // 2): text_pos + int(max_seq_len // 2)]
        else:
            s += text
    else:
        if title_pos < 0:
            text = text[text_pos - int(max_seq_len // 2): text_pos + int(max_seq_len // 2)]

        if title not in text:
            s += title + ' '
        s += text
    return s


def data_process():
    df = pd.read_csv('./trainsample/Train_Data.csv')

    # 文本预处理
    df.fillna({'title': '', 'text': '', 'entity': '', 'key_entity': ''}, inplace=True)
    df['title'] = df['title'].map(lambda x: text_preprocess(x))
    df['text'] = df['text'].map(lambda x: text_preprocess(x))

    # 过滤实体列表
    df['org_entity'] = df['entity']
    df['merge_text'] = df.apply(lambda x: merge_text1(x.title, x.text), axis=1)
    df['entity'] = df['entity'].map(lambda x: full2half(x))
    df['entity'] = df.apply(lambda x: filter_entity(x.entity, x.merge_text), axis=1)
    df['key_entity'] = df['key_entity'].map(lambda x: full2half(x))
    # 长度统计
    df['l_title'] = df['title'].map(lambda x: len(x))
    df['l_text'] = df['text'].map(lambda x: len(x))
    df['l_entity'] = df['entity'].map(lambda x: len(x))
    df['l_kentity'] = df['key_entity'].map(lambda x: len(x.split(';')) if x else 0)
    df['l_other'] = df['l_entity'] - df['l_kentity']

    # 过滤无实体文本
    df = df.loc[df['l_entity'] != 0, :]  # 过滤没有实体的样本
    print(df.describe(percentiles=[0.75, 0.95, 0.99]))
    return df


def mask_other_entity(text, other_entity):
    for i in other_entity:
        text= text.replace(i, '[O]')
    return text


def tag_pred_entity(text, entity):
    text = text.replace(entity, '[E]' + entity + '[E]')
    return text


def task_format1(df):
    """
    双输入文本+二分类任务
    text1：实体
    text2：title+text
    其他伴随实体都用[O]进行替换
    """
    # 把多个实体展开成多条样本
    df['single_entity'] = df['entity']
    df = df.explode('single_entity').reset_index(drop=True)
    # label = negative & 实体为核心实体
    if 'negative' in df.columns:
        df['label'] = df.apply(lambda x: 1 if x.single_entity in x.key_entity and x.negative else 0, axis=1)

    # 定位实体在标题和文本中的位置
    df['title_pos'] = df.apply(lambda x: find_pos(x.title, x.single_entity), axis=1)
    df['text_pos'] = df.apply(lambda x: find_pos(x.text, x.single_entity), axis=1)

    # 针对前510没有出现实体的，截取实体出现位置前300+后200
    df['corpus'] = df.apply(lambda x: merge_text2(x.title, x.text, x.single_entity, x.title_pos, x.text_pos),
                            axis=1)

    # 定位伴随实体 & 特殊token替换伴随实体
    df['other_entity'] = df.apply(lambda x: [i for i in x.entity if i != x.single_entity], axis=1)
    df['corpus'] = df.apply(lambda x: mask_other_entity(x.corpus, x.other_entity), axis=1)
    return df


def task_format2(df):
    """
    单输入文本+二分类任务
    text：title+text
    待预测实体用’[E]‘在实体的左右边界进行标记，其他伴随实体都用[O]进行替换
    """
    df['single_entity'] = df['entity']
    df = df.explode('single_entity').reset_index(drop=True)
    # label = negative & 实体为核心实体
    if 'negative' in df.columns:
        df['label'] = df.apply(lambda x: 1 if x.single_entity in x.key_entity and x.negative else 0, axis=1)

    # 定位实体在标题和文本中的位置
    df['title_pos'] = df.apply(lambda x: find_pos(x.title, x.single_entity), axis=1)
    df['text_pos'] = df.apply(lambda x: find_pos(x.text, x.single_entity), axis=1)
    # 针对前510没有出现实体的，截取实体出现位置前300+后200
    df['corpus'] = df.apply(lambda x: merge_text2(x.title, x.text, x.single_entity, x.title_pos, x.text_pos),
                            axis=1)

    # 添加实体标记
    df['corpus'] = df.apply(lambda x: tag_pred_entity(x.corpus, x.single_entity), axis=1)

    # 定位伴随实体 & 特殊token替换伴随实体
    df['other_entity'] = df.apply(lambda x: [i for i in x.entity if i != x.single_entity], axis=1)
    df['corpus'] = df.apply(lambda x: mask_other_entity(x.corpus, x.other_entity), axis=1)

    return df


def task_format3(df):
    """
    双输入文本+二分类任务
    text1：其他实体拼接
    text2：title+text
    其他伴随实体都用[O]进行替换
    待预测实体用’[E]‘在实体的左右边界进行标记
    """
    df['single_entity'] = df['entity']
    df = df.explode('single_entity').reset_index(drop=True)
    # label = negative & 实体为核心实体
    if 'negative' in df.columns:
        df['label'] = df.apply(lambda x: 1 if x.single_entity in x.key_entity and x.negative else 0, axis=1)

    # 定位实体在标题和文本中的位置
    df['title_pos'] = df.apply(lambda x: find_pos(x.title, x.single_entity), axis=1)
    df['text_pos'] = df.apply(lambda x: find_pos(x.text, x.single_entity), axis=1)
    # 针对前510没有出现实体的，截取实体出现位置前300+后200
    df['corpus'] = df.apply(lambda x: merge_text2(x.title, x.text, x.single_entity, x.title_pos, x.text_pos),
                            axis=1)

    # 添加实体标记
    df['corpus'] = df.apply(lambda x: tag_pred_entity(x.corpus, x.single_entity), axis=1)

    # 定位伴随实体 & 特殊token替换伴随实体
    df['other_entity'] = df.apply(lambda x: [i for i in x.entity if i != x.single_entity], axis=1)
    df['other_entity'] = df['other_entity'].map(lambda x: ' '.join(x))
    return df


if __name__ == '__main__':
    from src.dataset.converter import single_text, double_text
    data_dir = './trainsample'

    df = data_process()
    df1 = task_format1(df)
    train, valid = train_test_split(df1, test_size= 0.2, random_state=1234)
    double_text(train['id'], train['single_entity'], train['corpus'], train['label'], data_dir, 'train1')
    double_text(valid['id'], valid['single_entity'], valid['corpus'], valid['label'], data_dir, 'valid1')

    df2 = task_format2(df)
    train, valid = train_test_split(df2, test_size= 0.2, random_state=1234)
    single_text(train['id'], train['corpus'], train['label'], data_dir, 'train2')
    single_text(valid['id', ], valid['corpus'], valid['label'], data_dir, 'valid2')

    df3 = task_format3(df)
    train, valid = train_test_split(df3, test_size= 0.2, random_state=1234)
    double_text(train['id'],train['other_entity'], train['corpus'], train['label'], data_dir, 'train3')
    double_text(valid['id'], valid['other_entity'], valid['corpus'], valid['label'], data_dir, 'valid3')