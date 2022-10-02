# -*-coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
from src.preprocess.str_utils import full2half, get_useless_chars

fix_false_ent = {
    'af86add3': '光线传媒',
    '675f27db': '乐学·乐享·乐于成长',
    '664602fe': '五星基金;华安策略',
    '72403375': 'GreatProfitGlobalLimited',
    'd1497ac7': 'missing',
    'e90fdc96': '觅信dec',
    '72026fe3': '欧莱雅;嗨团团购',
    'b6c50768': 'DaVinciResolve',
    'bd77a98c': '玖富普惠;你我贷;上海你我贷互联网金融信息服务有限公司;宜人贷;微贷网',
    'd4066c42': '夏商风信子;时福全球购;恒优国际;宝象商务中心;金淘惠源;沃洋优品;酩悦酒业;欧颂酒业;优传保税;跑街;欧食安;海捣网;酒龙网;E境·国际生活体验城;跨惠通;跑街',
    '2e0e3ce5': 'missing',
    '140e24ed': 'missing',
    'db5dd2d1': '苏宁体育;苏宁文创;苏宁影城;苏宁易购;红孩子;苏宁小店;零售云',
    'b541d306': '19号谷仓手工吐司;19号谷仓',
    '398cd900': 'missing',
    'f6df2f3c': '华润商业;平安不动产;平安磐海汇富;中海海运资产;合享投资;中华企业;金投基金;中星集团;金丰投资;上海古北集团',
    '1e9009e7': "东易家装",
    '38a05aee': '北京钦一投资;上海三玺资产;深圳升龙基金;南京安赐投资;上海懋融;嘉兴观复投资管理有限公司;浙江谢志宇控股集团有限公司;杭州凯蓝汽车租赁有限公司',
    '7d76b0d7': '雅布力',
    '183fa7c9': 'missing',
    '5ec87d25': 'missing',
    '5d440622': 'missing',
    'd726a20a': 'brt房地产信托',
    'f0d49b34': 'missing',
    '269c5c49': '绿都影院',
    '59ad1655': '钱镜',
    '1bf18058': '高胜投资;投行配资;九鼎金策;香港信诚资产;中鑫建投;帝锋金业;向上金服;银丰配资;粤友钱;策马财经;盈龙策略',
    '3dc1d351': 'hes和氏币;h币',
    '586f7c62': 'missing',
    'fd8bc768': '合创',
    '0b9d754c': 'atc国际期货;香港恒利金业;嘉信金服',
    '929424f6': '寒武创投;熠美投资',
    '0dd5634f': '汇商传媒;FernGroupNZLimited',
    'fdadf73b': '洛安期货;昶胜国际;bkb数字货币;中恒策略;mbgmarkets',
    '7a35ebf4': 'missing',
    'f130c150': '东方花旗证券;东方花旗证券东方花旗证券有限公司;;东方证券股份有限公司;花旗环球金融（亚洲）有限公司',
    'df74b445': '博旅',
    '8d855f22': '黑鱼;有钱乐;满分优享;周转宝盒;同程借钱;沃克金服;花不缺',
    'cc1f41d7': '金网安泰;信雅达',
    '09dbcb15': '亚马逊',
    '233da315': '海拓环球',
    '0810f0d6': '东霖国际集团;quantlab(ql)量化券商',
    '55f8f9b4': '灵资本;道生资本;小智投资',
    '8d6bb6fb': '中资信汇投资',
    'c94c6719': '海航集团;航海创新',
    '44430728': 'missing',
    '2f747cbf': '北斗和正科技公司;贵州吉祥数贸公司',
    'cfa195a1': 'missing',
    'af661c44': '领航国际资本',
    '735e1fea': '比特币;莱特币;无限币;夸克币;泽塔币;烧烤币;隐形金条',
    'e48cc662': 'missing',
    'f8fed1f5': '抖音;快手',
    'f5046912': 'missing',
    '856e9d9d': '小九花花',
    '846d7fd4': 'cmsl;四川创梦森林软件科技有限公司',
    '00d453ce': 'missing',
    '70591f6c': 'missing',
    '204baee9': 'BithumbGlobal',
    'ebb3143': 'missing',
    '4b7cd57e': 'missing',
    '3c7be5e8': 'missing',
    '09204373': '以太云;ETY',
    'f12cf11b': '蘑菇街;飞猪旅行;侠侣联盟;厦门侠网旅游服务有限公司;厦门侠网旅游服务有限公司',
    'fe3c42ea': 'dlc(dolahcoin);多拉币',
    '2e3c9394': 'missing',
    '63599fd9': 'missing',
    'a117692f': '广药集团',
    '56023942': 'missing',
    'ef9495df': 'missing',
    'f47e2948': '深圳市善心汇文化传播有限公司;善心汇',
    '4c694de9': '红宝石娱乐;通宝娱乐;微云秀',
    '21b1a448': '点牛股',
    'b7f20246': 'missing',
    '17da7b21': 'missing',
    '3cf728e2': '金证股份;普康视',
    'eb7d85a8': 'missing',
    '67351350': 'missing'
}


def fix_entity(entity):
    # 实体清洗
    if not entity:
        return entity
    # 剔除2个以上的？
    entity = re.sub('\?{2,}', '', entity)
    # 消重
    entity = sorted(set(entity.split(';')))
    # 过滤空实体
    entity = filter(None, entity)
    # 剔除实体前后空格
    entity = ';'.join(map(str.strip, entity))
    # full2half
    entity = full2half(entity)
    # tolower
    entity = entity.lower()
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
                pos = list(pos.span())  # 改成左闭右闭的
                pos_list.append(['FIN'] + [pos[0], pos[1] - 1])
        except Exception as e:
            print(e, entities)
    return pos_list


def false_ent(text, entities):
    # 定位实体全部识别错误的样本并剔除
    if not entities:
        return False
    if entities == 'missing':
        return True
    flag = any((ent in text for ent in entities.split(';')))
    return not flag


def text_preprocess(s, useless_chars):
    # 注意这里做了大写转小写的转换，如果是比赛需要在最终提交的时候进行还原
    if not s:
        return s

    s = full2half(s)
    # 图片
    s = re.sub('\{IMG:.?.?.?\}', '', s)
    # http
    s = re.sub('(https?|ftp|file|www)(:\/{2}|\.)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', s)  # 过滤网址
    s = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', s).replace("()", "")  # 过滤www开头的m
    # http tag
    s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签

    s = re.sub('^\?+', '', s)  # 问号开头直接删除
    s = re.sub('\?{2,}', ' ', s)  # 中间出现多余1个问号用空格替代

    s = re.sub('^#', ' ', s)  # 井号开头直接删除
    s = re.sub('#+', ' ', s)  # 井号在中间用空格分隔

    s = re.sub(re.compile('微信[:：]?[a-zA-Z0-9]+'), ' ', s)  # 微信
    s = re.sub(re.compile('(\d{4}-\d{2}-\d{2})|(\d{2}:\d{2}:\d{2})'), ' ', s)  # 时间

    # s = re.sub(mention_handler.re_pattern, ' ', s)  # @
    s = s.lower()
    # fitler useless chars
    for uc in useless_chars:
        s = s.replace(uc, '')
    # remove space in text to avoid tokenizer mismatch
    s = re.sub(r'\s{1,}', '', s)
    return s


def split_text(title, text, max_seq_len):
    """
    text按标点符号超过长度后进行split，
    """
    ## 满足长度的直接返回
    if not title or title in text:
        title = ''
        if len(text) <= max_seq_len:
            return [text]
    else:
        if len(text) + len(title) <= max_seq_len:
            return [title + text]
    ## 不满足长度的按句子进行拆分后得到多段文本
    sentences = hierarchy_text_split(text, max_seq_len)
    corpus = []
    tmp = title
    for i in range(len(sentences)):
        if len(tmp) <= max_seq_len and len(tmp) + len(sentences[i]) > max_seq_len:
            corpus.append(tmp.strip())
            tmp = sentences[i]
        else:
            tmp += sentences[i]
    if tmp:
        corpus.append(tmp.strip())
    return corpus


def hierarchy_text_split(text, max_seq_len):
    sentences = []
    tmp = []
    # 优先使用句子分割
    for s in text:
        tmp.append(s)
        if s in {'。', '！', '!', '？', '?'}:
            if len(tmp) <= max_seq_len:
                sentences.append(''.join(tmp))
            else:
                new_sentences = []
                new_tmp = []
                for ss in tmp:
                    new_tmp.append(ss)
                    if ss in {',', '，', ':', '：'}:
                        l = len(new_tmp)
                        if l > max_seq_len:
                            # 存在少量样本中间无分隔符
                            new_sentences.append(''.join(new_tmp[:l // 2]))
                            new_sentences.append(''.join(new_tmp[(l // 2):]))
                        else:
                            new_sentences.append(''.join(new_tmp))
                        new_tmp = []
                if new_tmp:
                    l = len(new_tmp)
                    if l > max_seq_len:
                        # 存在少量样本中间无分隔符
                        new_sentences.append(''.join(new_tmp[:l // 2]))
                        new_sentences.append(''.join(new_tmp[(l // 2):]))
                    else:
                        new_sentences.append(''.join(new_tmp))
                sentences += new_sentences
            tmp = []

    if len(tmp) > 0:
        if len(tmp) <= max_seq_len:
            sentences.append(''.join(tmp))
        else:
            new_sentences = []
            new_tmp = []
            for ss in tmp:
                new_tmp.append(ss)
                if ss in {',', '，', ':', '：', ')', '）'}:
                    l = len(new_tmp)
                    if l > max_seq_len:
                        # 存在少量样本中间无分隔符
                        new_sentences.append(''.join(new_tmp[:l // 2]))
                        new_sentences.append(''.join(new_tmp[(l // 2):]))
                    else:
                        new_sentences.append(''.join(new_tmp))
                    new_tmp = []
            if new_tmp:
                l = len(new_tmp)
                if l > max_seq_len:
                    # 存在少量样本中间无分隔符
                    new_sentences.append(''.join(new_tmp[:l // 2]))
                    new_sentences.append(''.join(new_tmp[(l // 2):]))
                else:
                    new_sentences.append(''.join(new_tmp))
            sentences += new_sentences
    return sentences


def data_process(file_name='./trainsample/Train_Data.csv'):
    df = pd.read_csv(file_name)
    df.fillna({'text': '', 'title': '', 'unknownEntities': ''}, inplace=True)

    # 替换错误实体
    df['fix_entity'] = df['id'].map(lambda x: fix_false_ent.get(x, ''))
    df['unknownEntities'] = df.apply(lambda x: x.fix_entity if x.fix_entity else x.unknownEntities, axis=1)

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
    df = df.loc[~(df['entities'].map(lambda x: len(x) > 0) & df['ent_pos'].map(lambda x: len(x) == 0)), :]
    print(f'切分样本后总共样本数{df.shape[0]}')
    return df


if __name__ == '__main__':
    print(hierarchy_text_split('今天,天气正好。', 4))
    print(hierarchy_text_split('今天，天气，正好', 6))
    print(hierarchy_text_split('今天,天气正好!特别好呀，特别好。', 8))
