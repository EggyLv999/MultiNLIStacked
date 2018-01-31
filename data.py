from __future__ import print_function, division

import numpy as np

import util

SHIFT=0
REDUCE=1

def parse_snli_tree(t):
    """Parse linearized tree in the SNLI format

    Returns a list of words and shift-reduce actions
    """
    words = []
    actions =[]
    stack = []
    word_pointer, action_pointer = 0, 0
    for token in t.split():
        if token == '(':
            stack.append((word_pointer, action_pointer))
            continue
        elif token == ')':
            actions.append(REDUCE)
            action_pointer+=1
        else:
            words.append(token)
            word_pointer += 1
            actions.append(SHIFT)
            action_pointer += 1
    assert 2 * len(words) - 1 == len(actions)
    return words, actions


def load_snli(filename, multi_labels=False, max_length=150):
    """Load a file in the SNLI format"""
    trees = util.loadtxt(filename)[1:]
    sentences, actions, labels, genres = [], [], [], []
    for t in trees:
        fields = t.split('\t')
        # Retrieve label
        if multi_labels:
            label = fields[10:]
        else:
            label = fields[0]

        # If label is blank, ignore sample
        if label == '-' or ('' in label and isinstance(label, list)):
            continue

        genre = fields[9]
        # Parse the two sentences
        t1 = fields[1].lower()
        t2 = fields[2].lower()
        w1, a1 = parse_snli_tree(t1)
        w2, a2 = parse_snli_tree(t2)
        # Ignore sentences if they are too long
        if max_length > 0 and len(w1) > max_length:
            continue
        # Ignore sentences that don't have a parse tree
        # if len(a1) == 1 or len(a2) == 1:
        #    continue
        sentences.append((w1, w2))
        actions.append((a1, a2))
        labels.append(label)
        genres.append(genre)
    return sentences, actions, labels, genres

def load_mnli_unlabeled(filename):
    trees = util.loadtxt(filename)[1:]
    sentences, actions, ids = [], [], []
    for t in trees:
        fields = t.split('\t')
        # Retrieve pairid
        pairid = fields[8]
        # Parse the two sentences
        t1 = fields[1].lower()
        t2 = fields[2].lower()
        w1, a1 = parse_snli_tree(t1)
        w2, a2 = parse_snli_tree(t2)
        sentences.append((w1, w2))
        actions.append((a1, a2))
        ids.append(pairid)
    return sentences, actions, ids

def shuffle_multinli(full_size, percent_snli=15):
    """This shuffles the training data such that at each epoch we hae the fully shuffled MultiNLI + some percent of the snli"""
    indices = np.arange(full_size, dtype=int)
    # There are 392702 multinli samples and 550152 SNLI samples
    multinli = indices[:392702]
    snli = indices[392702:]

    if percent_snli > 0:
        np.random.shuffle(snli)
        size_snli = int(percent_snli / 100 * len(snli))
        multinli = np.concatenate([multinli, snli[:size_snli]])

    np.random.shuffle(multinli)
    return multinli 


