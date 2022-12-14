# -*-coding:utf-8 -*-
"""
    基于词频，互信息，和左右交叉熵的新词发现算法
"""
from itertools import chain
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from ct.utils.str_utils import *
import math
import jieba

# load dict
jieba.initialize()


def text_process(sentence):
    """
    1. 字符粒度tokenize
    2. 过滤数字
    3. 遇到标点符号跳过并返回token list
    3. 返回list of list
    """
    text = full2half(sentence)
    output = []
    tmp = []
    for i in text:
        if punctuation_handler.check(i):
            if tmp:
                output.append(tmp)
            tmp = []
            continue
        if i == '.':
            continue

        if i.strip() and not i.isdigit():
            tmp.append(i)
        else:
            if tmp:
                output.append(tmp)
            tmp = []
    if tmp:
        output.append(tmp)
    return output


class TreeNode(object):
    def __init__(self, char=None):
        self.char = char
        self.children = {}
        self.freq = 0
        self.children_freq = defaultdict(int)

    def add_surrounding(self, c):
        self.children_freq[c] += 1

    def insert(self, c):
        self.children_freq[c] += 1
        if c in self.children:
            return self.children[c]
        else:
            self.children[c] = TreeNode(c)
            return self.children[c]

    def search(self, c):
        return self.children.get(c, None)


class Trie:
    def __init__(self):
        self.root = TreeNode()
        self.total_len = 0

    def insert(self, word):
        node = self.root
        for c in word:
            self.total_len += 1
            node.freq += 1
            node = node.insert(c)
        node.freq += 1
        return node

    def start_with(self, word):
        node = self.root
        for c in word:
            node = node.search(c)
            if not node:
                return None
        return node

    def get_ngram(self, min_depth=2):
        # get all ngram in Trie Tree with n>=min_depth
        ngrams = []

        def dfs(node, depth, path):
            if depth >= min_depth:
                ngrams.append(path)

            if not node.children:
                return

            for c in node.children.values():
                dfs(c, depth + 1, path + [c.char])

        for node in self.root.children.values():
            dfs(node, 1, [node.char])
        return ngrams

    def get_freq(self, word):
        node = self.start_with(word)
        if node is None:
            return 0
        else:
            return node.freq

    @staticmethod
    def calc_entropy(freq_list):
        total = sum(freq_list)
        prob = map(lambda x: x / total, freq_list)
        entropy = -1 * sum(map(lambda x: x * math.log2(x), prob))
        return entropy

    def get_entropy(self, word):
        """
        calculate entropy of following char: sum(-p*log(p)) the bigger the better，special cases
        - only apper at begining or at end: return None
        - only has 1 neighbour: return 0
        - else return entropy
        """
        node = self.start_with(word)
        if not node or not node.children_freq:
            return None
        elif len(node.children_freq) == 1:
            return 0
        else:
            return self.calc_entropy(node.children_freq.values())

    def get_pmi(self, word):
        """
        calcualte min pmi of all combination of word: p(x,y)/p(x) * p(y), the bigger the better, special case
        - for ngram bigger than 2, return minimal of all combination
        """
        min_pmi = np.inf
        for i in range(1, len(word)):
            pmi = self.get_freq(word) * self.total_len / (self.get_freq(word[:i]) * self.get_freq(word[i:]))
            min_pmi = min(min_pmi, pmi)
        return min_pmi


class NewWordDetection(object):
    def __init__(self, ngram, min_freq, min_pmi, min_entropy):
        self.ftree = Trie()  # forward trie tree
        self.btree = Trie()  # backward trie tree
        self.ngram = ngram
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.min_entropy = min_entropy
        self.corpus = None

    def build_tree(self, sentence_list):
        self.corpus = list(chain(*map(text_process, sentence_list)))
        for s in tqdm(self.corpus):
            l = len(s)

            for i in range(l):
                # iter to the last token
                fnode = self.ftree.insert(s[i:i + self.ngram])
                bnode = self.btree.insert(s[max(i-self.ngram+1, 0): i+1][::-1])

                # add surrounding for end token words
                if i + self.ngram < l:
                    fnode.add_surrounding(s[i + self.ngram])
                if i +1 -self.ngram>0:
                    bnode.add_surrounding(s[i - 1])

    def calc_score(self):
        self.candidates = []
        for word in tqdm(self.ftree.get_ngram()):
            freq = self.ftree.get_freq(word)
            if freq > self.min_freq:
                pmi = self.ftree.get_pmi(word)
                if pmi > self.min_pmi:
                    left_entropy = self.btree.get_entropy(word[::-1])
                    right_entropy = self.ftree.get_entropy(word)
                    if left_entropy is None and right_entropy is None:
                        entropy = np.inf
                    elif left_entropy is None:
                        entropy = right_entropy
                    elif right_entropy is None:
                        entropy = left_entropy
                    else:
                        entropy = min(left_entropy, right_entropy)
                    if entropy > self.min_entropy:
                        self.candidates.append({'word': ''.join(word),
                                                'freq': freq,
                                                'pmi': pmi,
                                                'l_entropy': left_entropy,
                                                'r_entropy': right_entropy,
                                                'entropy': entropy})
        self.candidates = sorted(self.candidates, key=lambda x: x['freq'], reverse=True)

    def get_topk(self, topk=None, new_only=True):
        # filter existing word in jieba default dict
        candidates = []
        if new_only:
            for i in self.candidates:
                if jieba.get_FREQ(i['word']):
                    continue
                candidates.append(i)
        if topk is None:
            return candidates
        else:
            return candidates[:topk]
