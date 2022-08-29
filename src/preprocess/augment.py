# -*-coding:utf-8 -*-

from src.preprocess.str_utils import *
import random


class Augmenter(object):
    def __init__(self, tokenizer, max_sample, prob, filters):
        self.max_sample = max_sample
        self.prob = prob
        self.filters = filters
        self.tokenizer = tokenizer

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

    def get_aug_index(self, tokens):
        index = []
        for i, t in enumerate(tokens):
            if any(f.check(t) for f in self.filters):
                continue
            index.append(i)
        return index

    def augment(self, text):
        max_retry = 2
        result = set()  # only keep non-dupulicate
        for _ in range(self.max_sample * max_retry):
            aug_text = self.action(text)
            if self._data_check(aug_text):
                result.add(' '.join(aug_text))

        if len(result) > self.max_sample:
            return random.sample(result, self.max_sample)
        else:
            return result


class W2vSynonymous(Augmenter):
    def __init__(self, tokenizer, max_sample=3, prob=0.1,
                 filters=(stop_word_handler, punctuation_handler, emoji_handler), topn=10):
        super(W2vSynonymous, self).__init__(tokenizer, max_sample, prob, filters)
        self.topn = topn

    def gen_synom(self, word):
        if random.random() < self.prob:
            try:
                nn = self.tokenizer.model.wv.most_similar(word, topn=self.topn)
                return random.choice(nn)[0]
            except:
                return None
        else:
            return None

    def action(self, text):
        words = self.tokenizer.tokenize(text)
        index = self.get_aug_index(words)
        flag = False
        for i in index:
            synom = self.gen_synom(words[i])
            if synom:
                flag = True
                words[i] = synom
        if flag:
            return words
        else:
            return None


class Wordnetsynonymous(Augmenter):
    def __init__(self, tokenizer, max_sample=3, prob=0.1,
                 filters=(stop_word_handler, punctuation_handler, emoji_handler)):
        super(Wordnetsynonymous, self).__init__(tokenizer, max_sample, prob, filters)
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
        words = self.tokenizer.tokenize(text)
        index = self.get_aug_index(words)
        flag = False
        for i in index:
            synom = self.gen_synom(words[i])
            if synom:
                flag = True
                words[i] = synom
        if flag:
            return words
        else:
            return None


class WordSwap(Augmenter):
    def __init__(self, tokenizer, max_sample=3, prob=0.1,
                 filters=(stop_word_handler, punctuation_handler, emoji_handler)):
        super(WordSwap, self).__init__(tokenizer, max_sample, prob, filters)

    def get_swap_pos(self, left, right):
        if left>right or random.random() >self.prob:
            return left-1
        else:
            return random.randint(left, right)

    def action(self, text):
        new_sample = []
        words = self.tokenizer.tokenize(text)
        index = self.get_aug_index(words)
        l = len(words)
        for i in index:
            pos = self.get_swap_pos(i + 1, l-1)
            words[i], words[pos] = words[pos], words[i]
            new_sample.append(words[i])
        return new_sample


class WordDelete(Augmenter):
    def __init__(self, tokenizer, max_sample=3, prob=0.1,
                 filters=(stop_word_handler, punctuation_handler, emoji_handler)):
        super(WordDelete, self).__init__(tokenizer, max_sample, prob, filters)

    def action(self, text):
        new_sample = []
        words = self.tokenizer.tokenize(text)
        index = self.get_aug_index(words)
        for i in range(len(words)):
            if i in index and random.random()< self.prob:
                continue
            new_sample.append(words[i])
        return new_sample
