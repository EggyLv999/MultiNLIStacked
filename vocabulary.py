from __future__ import print_function, division

import pickle
import numpy as np
from collections import defaultdict

import sys

UNK_TOKEN = '<unk>'

def zero():
    return 0

def dd():
    return defaultdict(zero)


def reverse_dic(dic):
    """Get the inverse mapping from an injective dictionary"""
    rev_dic = dict()
    for k, v in dic.items():
        rev_dic[v] = k
    return rev_dic


def make_dic(data, max_size=100000, min_freq=-1):
    """Make dictionary words -> indices based on frequency"""
    dic = dict  ()
    freqs = defaultdict(zero)
    dic[UNK_TOKEN] = 0
    for sent in data:
        for word in sent:
            freqs[word] += 1
    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(max_size, len(sorted_words))):
        word, freq = sorted_words[i]
        if freq < min_freq:
            continue
        dic[word] = len(dic)
    return dic, reverse_dic(dic)


class Vocabulary(object):
    """The vocabulary class handles the transition words/ids among other things"""

    def __init__(self):
        self.w2id = {}   # Word to id (source)
        self.id2w = {}   # Id to word (source)
        self.classes = {}

        self.unigrams = np.zeros(1) # Unigram distribution

    def sents_to_ids(self, sents):
        """Convert sequences of words to sequences of ids"""
        return np.asarray([self.sent_to_ids(s) for s in sents], dtype=list)

    def sent_to_ids(self, sent):
        """Convert sequence of words to sequence of ids"""
        x = [self.w2id[w] if w in self.w2id else self.w2id[UNK_TOKEN] for w in sent]
        return x

    def ids_to_sent(self, x):
        """Convert sequence of ids to sequence of ids"""
        sent = [self.id2w[i] for i in x]
        return sent

    def translate_literal_labels(self, labels):
        if isinstance(labels[0], list):
            # If we're given a list of labels, output the distribution
            f = self.labels_to_distribution
        else:
            # Otherwise just the label
            f = self.label_to_ids
        return np.asarray([f(label) for label in labels])

    def label_to_ids(self, label):
        return self.classes[label]

    def labels_to_distribution(self, labels):
        ret = np.zeros(len(self.classes))
        for l in labels:
            ret[self.classes[l]] += 1
        ret /= ret.sum()
        return ret

    def init(self, data, labels, opt):
        # Read dictionaries from training files
        self.w2id, self.id2w = make_dic(data,
                                        max_size=opt.vocab_size,
                                        min_freq=opt.min_freq)
        if opt.multi_labels:
            labels_list = [set(l) for l in labels]
            labels = set.union(*labels_list)
        else:
            labels = set(labels)
        self.classes = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        assert len(labels) == len(self.classes)
        # Estimate unigram distribution
        self.unigrams = self.compute_unigrams(data, laplace_smoothing=opt.laplace_smoothing)

    def compute_unigrams(self, corpus, laplace_smoothing=0.0):
        voc_size = len(self.id2w)
        unigrams = np.zeros(voc_size) + laplace_smoothing
        for sent in corpus:
            for w_id in self.sent_to_ids(sent):
                unigrams[w_id] += 1
        unigrams /= unigrams.sum()
        return unigrams

def save_vocab(filename, vocab):
    """Save vocabulary"""
    print('Saving vocabulary to file %s' % filename)
    with open(filename, 'wb+') as f:
        pickle.dump(vocab, f)


def load_vocab(filename):
    print('Reading vocabulary from file %s' % filename)
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


