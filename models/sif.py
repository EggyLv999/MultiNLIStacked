from __future__ import print_function, division

import sys
import numpy as np
import dynet as dy


class SIF(object):

    def __init__(self, lex, opt):

        self.pc = dy.ParameterCollection()

        # Hyper parameters
        self.embed_dim = opt.embed_dim
        # Unigram weights
        if opt.sif_a > 0:
            weights = (opt.sif_a / (opt.sif_a + lex.unigrams)).reshape(-1, 1)
        else:
            weights = np.ones((len(lex.unigrams), 1))
        self.W = self.pc.lookup_parameters_from_numpy(weights, name='weights')
        # Word embeddings
        self.E = self.pc.add_lookup_parameters((len(lex.id2w), self.embed_dim), name='E')
        # Load pretrained word embeddings
        if opt.pretrained_wembs is not None:
            print('Initializing word embeddings from %s' % opt.pretrained_wembs)
            sys.stdout.flush()
            word_vectors = np.load(opt.pretrained_wembs)
            self.E.init_from_array(word_vectors)



    def compose(self, x, a, update=True, test=False, NT=False):
        """Compose a sentence into a vector"""
        # Retrieve word vectors
        word_vectors =dy.concatenate_cols([dy.lookup(self.E, w, update=True) for w in x])
        # Retrieve word weights
        word_weights = dy.concatenate([dy.lookup(self.W, w, update=False) for w in x])
        # Compute average
        Z = dy.sum_elems(word_weights)
        h = dy.cdiv(word_vectors * word_weights, Z)
        # Return representation
        return h

    def compare(self, x, y, ax, ay, update=True, test=False):
        hx = self.compose(x, ax, update, test)
        hy = self.compose(y, ay, update, test)
        delta = dy.concatenate([hx, hy, dy.cmult(hx, hy), dy.abs(hx - hy)])
        return delta

    def compare_batch(self, words, actions, update=True, test=False):
        deltas = [self.compare(x, y, None, None, update=update, test=test) for x, y in words]
        return dy.concatenate_to_batch(deltas)


