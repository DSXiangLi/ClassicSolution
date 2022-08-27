# -*-coding:utf-8 -*-

from src.preprocess.str_utils import *
import random
from concurrent.futures import ThreadPoolExecutor


class Augmenter(object):
    """
    Action: Delete, Swap, Substitute
    Granularity: char, word, entity, sentence
    """

    def __init__(self, min_sample, max_sample, prob):
        self.min_sample = min_sample
        self.max_sample = max_sample
        self.prob = prob

    def action(self, text):
        """
        Core Augment action
        """
        raise NotImplementedError

    @staticmethod
    def _data_check(data):
        if not data or len(data) == 0:
            return False
        else:
            return True

    def _param_check(self):
        pass

    def augment(self, data, n_thread=4):
        """

        :param augment:
        :param data:
        :param n:
        :param n_thead:
        :return:
        """
        min_output = len(data) * self.min_sample
        max_output = len(data) * self.max_sample  # 默认增强样本<=原始样本
        max_retry = 3
        result = set()  # only keep non-dupulicate
        for _ in range(max_retry):
            with ThreadPoolExecutor(n_thread) as executor:
                for aug_data in executor.map(self.action, data):
                    if self._data_check(aug_data):
                        result.add(aug_data)
            if len(result) > min_output:
                break
        if len(result) > max_output:
            return random.sample(result, max_output)
        else:
            return result


class WordAugmenter(Augmenter):
    def __init__(self, min_sample, max_sample, prob, tokenizer):
        super(WordAugmenter, self).__init__(min_sample, max_sample, prob)
        self.filters = [stop_word_handler, punctuation_handler, emoji_handler]
        self.tokenizer = tokenizer

    def get_aug_index(self, tokens):
        index = set()
        for i, t in enumerate(tokens):
            if any(f.check(t) for f in self.filters):
                continue
            index.add(i)
        return index


class W2vSynomous(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, tokenizer, topn=10):
        super(W2vSynomous, self).__init__(min_sample, max_sample, prob, tokenizer)
        self.topn = topn

    def gen_synom(self, word):
        if random.random() < self.prob:
            try:
                nn = self.tokenizer.model.most_similar(word, topn=self.topn)
                return random.choice(nn)[0]
            except:
                return None
        else:
            return None

    def action(self, text):
        new_sample = []
        words = self.tokenizer.tokenize(text)
        flag = False
        for i, t in enumerate(words):
            if i in self.get_aug_index(words):
                self.gen_synom(t)
                flag = True
            else:
                new_sample.append(t)
        if flag:
            return new_sample
        else:
            return None


class WordnetSynomous(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, tokenizer):
        super(WordnetSynomous, self).__init__(min_sample, max_sample, prob, tokenizer)
        self.wordnet = self.load('word_net.text')

    def load(self, file):
        wordnet = {}
        with open(file, 'r') as f:
            for line in f:
                line = line.strip().split(" ")
                if not line[0].endswith('='):
                    continue
                for i in range(1, len(line)):
                    wordnet[line[i]] = line[1:i] + line[(i + 1):]
        return wordnet

    def gen_synom(self, word):
        if word in self.wordnet and random.random() < self.prob:
            return random.choice(self.wordnet[word])
        else:
            return word

    def action(self, text):
        new_sample = []
        tokens = self.tokenizer.tokenize(text)
        flag = False
        for i, t in enumerate(tokens):
            if i in self.get_aug_index(tokens):
                self.gen_synom(t)
                flag = True
            else:
                new_sample.append(t)
        if flag:
            return tokens
        else:
            return None


class WordShuffle(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, tokenizer):
        super(WordShuffle, self).__init__(min_sample, max_sample, prob, tokenizer)

    def get_swap_pos(self, left, right):
        if random.random() < self.prob:
            return random.randint(left, right)
        else:
            return left - 1

    def action(self, text):
        new_sample = []
        tokens = self.tokenizer.tokenize(text)
        l = len(text)
        for i, t in enumerate(tokens):
            if i in self.get_aug_index(tokens):
                pos = self.get_swap_pos(i + 1, l - 1)
                tokens[i], tokens[pos] = tokens[pos], tokens[i]
            new_sample.append(tokens[i])
        return new_sample


class WordDelete(WordAugmenter):
    def __init__(self, min_sample, max_sample, prob, tokenizer):
        super(WordDelete, self).__init__(min_sample, max_sample, prob, tokenizer)

    def action(self, text):
        new_sample = []
        tokens = self.tokenizer.tokenize(text)
        for i, t in enumerate(tokens):
            if i in self.get_aug_index(tokens):
                if random.random()< self.prob:
                    continue
            new_sample.append(t)
        return new_sample
