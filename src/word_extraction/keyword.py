# -*-coding:utf-8 -*-
"""
    Keyword Extraction Method
    - TF-IDF
    - Bm25
    - TextRank
"""
import math
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
import jieba.posseg as pseg

from ct.utils.str_utils import *


def text_process(text):
    """
    preprocess of text
    1. full2half
    2. segmentation + pos tag
    3. filter punctuation, stop words, digit
    Return: list of words
    """
    text = full2half(text)
    word_list = pseg.dt.cut(text)
    output = []
    for i in word_list:
        if punctuation_handler.check(i.word):
            continue
        if stop_word_handler.check(i.word):
            continue
        if '..' in i.word:
            continue
        if i.word.strip() and not i.word.isdigit():
            output.append(i)
    return output


class KeywordExtractor(object):
    def __init__(self, min_freq, title_weight):
        self.min_freq = min_freq  # filter rare words
        self.title_weight = title_weight  # up weight the tf in title


class TfIdf(KeywordExtractor):
    def __init__(self, min_freq, title_weight):
        super(TfIdf, self).__init__(min_freq, title_weight)
        self.idf = defaultdict(int)
        self.med_idf = None

    def set_idf(self, doc_list):
        """
        Calculate IDF: IDF = log(total document / #docuemnt with word)
            doc_list: preprocessed word list
        """
        n_doc = 0
        for doc in doc_list:
            for word in set(map(lambda x: x.word, doc)):
                self.idf[word] += 1
            n_doc += 1

        # take log of the idf, to smooth the denominator
        for key, val in self.idf.items():
            self.idf[key] = math.log(n_doc / val)
        # for missing word ,use median IDF as default
        self.med_idf = np.median(list(self.idf.values()))

    def calc_keyword(self, content, title=None, topn=None):
        """
        Calculate TF： TF=(word freq/total word in docuemnt), upweighted the word in title
            title: sentence
            content: sentence
            topn: if topn is None return all words with weight
        """
        content = text_process(content)
        tf = Counter(content)
        total_words = sum(tf.values())

        if title is not None:
            title = text_process(title)
            tf_title = Counter(title)
            total_words += sum(tf_title.values())
            for word, freq in tf_title.items():
                tf[word] += freq * self.title_weight

        score = {}
        for word, freq in tf.items():
            # Filter word < min_freq and single character
            if freq > self.min_freq and len(word.word) > 1:
                score[word.word] = freq * self.idf.get(word, self.med_idf) / total_words

        if topn is None:
            return score
        else:
            output = sorted(score.items(), key=lambda x: x[1], reverse=True)
            output = output[:topn]
            return dict(output)


class Bm25(KeywordExtractor):
    def __init__(self, k, b, min_freq, title_weight):
        super(Bm25, self).__init__(min_freq, title_weight)
        self.k = k
        self.b = b
        self.idf = defaultdict(int)
        self.med_idf = None
        self.avg_dl = None

    def set_idf(self, doc_list):
        """
        Calculate IDF: IDF = log(total document / #docuemnt with word)
        """
        n_doc = 0
        doc_l = 0
        for doc in doc_list:
            for word in set(map(lambda x: x.word, doc)):
                self.idf[word] += 1
                doc_l += len(doc)
            n_doc += 1
        self.avg_dl = doc_l / n_doc

        # take log of the idf, to smooth the denominator
        for key, val in self.idf.items():
            self.idf[key] = math.log(n_doc / val)
        # for missing word ,use median IDF as default
        self.med_idf = np.median(list(self.idf.values()))

    def calc_keyword(self, content, title=None, topn=None):
        """
        Calculate TF： TF=(word freq/total word in docuemnt), upweighted the word in title
            title: sentence
            content: sentence
            topn: if topn is None return all words with weight
        """
        content = text_process(content)
        tf = Counter(content)
        total_words = sum(tf.values())

        if title is not None:
            title = text_process(title)
            tf_title = Counter(title)
            total_words += sum(tf_title.values())
            for word, freq in tf_title.items():
                tf[word] += freq * self.title_weight
        # calc score
        score = {}
        for word, freq in tf.items():
            # Filter word < min_freq and single character
            if freq > self.min_freq and len(word.word) > 1:
                score[word.word] = freq * self.idf.get(word, self.med_idf) * (self.k + 1) / (
                        freq + self.k * (1 - self.b + self.b * total_words / self.avg_dl))

        if topn is None:
            return score
        else:
            output = sorted(score.items(), key=lambda x: x[1], reverse=True)
            output = output[:topn]
            return dict(output)


class TextRank(KeywordExtractor):
    max_iter = 10
    d = 0.85
    threshold = 1e-3

    def __init__(self, window_size, min_freq, title_weight, allow_pos=('ns', 'n', 'vn', 'v')):
        super(TextRank, self).__init__(min_freq, title_weight)
        self.window_size = window_size
        self.allow_pos = allow_pos

    def _word_pair(self, word_list):
        filter_cond = lambda x: x.flag not in self.allow_pos or len(x.word) < 2
        word_pair = defaultdict(int)
        lt = len(word_list)
        for i, x1 in enumerate(word_list):
            if filter_cond(x1):
                continue
            for j in range(i + 1, i + self.window_size):
                if j >= lt:
                    break
                if filter_cond(word_list[j]):
                    continue
                word_pair[(x1.word, word_list[j].word)] += 1 * self.title_weight
        return word_pair

    def build_graph(self, content, title):
        self.G = nx.Graph()
        self.score = {}
        # build word pair count
        wp = self._word_pair(content)
        if title is not None:
            wp1 = self._word_pair(title)
            for key, val in wp1.items():
                wp[key] += val

        # add edges to graph
        for key, val in wp.items():
            self.G.add_edge(key[0], key[1], weight=val)

        # init node score
        for node in self.G.nodes():
            self.score[node] = 1 / len(self.G.edges())

    def train(self):
        for i in range(self.max_iter):
            max_delta = -1
            # iter all the nodes
            for key, val in sorted(self.score.items()):
                # iter all the neighbour of node i
                score = 0
                for j in self.G[key]:
                    score += self.score[j] / self.G.degree(j)
                score += (1 - self.d) + self.d * score
                # calculate max change in score
                max_delta = max(max_delta, abs(self.score[key] - score))
                self.score[key] = score

            if max_delta < self.threshold:
                break

        # min-max normalization
        min_score = min(self.score.values())
        max_score = max(self.score.values())
        for n, w in self.score.items():
            self.score[n] = (w - min_score) / (max_score - min_score)

    def calc_keyword(self, content, title=None, topn=None):
        content = text_process(content)
        if title is not None:
            title = text_process(title)
        self.build_graph(content, title)
        self.train()
        if topn is None:
            return self.score
        else:
            output = sorted(self.score.items(), key=lambda x: x[1], reverse=True)
            output = output[:topn]
            return dict(output)
