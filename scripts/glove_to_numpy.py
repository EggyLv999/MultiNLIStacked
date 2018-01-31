from __future__ import print_function, division

import sys
import numpy as np
import vocabulary

def make_glove_embbeding_matrix(filename, vocab, dim=100):
    """Make numpy array from word embeddings with the glove format and vocabulary object"""
    word_vectors = np.zeros((len(vocab.id2w), dim))
    with open(filename, 'r') as f:
        for l in f:
            fields = l.split()
            w = fields[0]
            if not w in vocab.w2id:
                continue
            vec = np.asarray(fields[1:], dtype=float)
            word_vectors[vocab.w2id[w]] = vec
    return word_vectors

if __name__ == '__main__':
    glove_file = sys.argv[1]
    vocab_file = sys.argv[2]
    out_file = sys.argv[3]
    dim = int(sys.argv[4])
    lexicon = vocabulary.load_vocab(vocab_file)
    word_vectors = make_glove_embbeding_matrix(glove_file, lexicon, dim)
    np.save(out_file, word_vectors)