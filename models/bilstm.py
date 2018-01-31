from __future__ import print_function, division

import sys
import numpy as np
import dynet as dy

from dynn import lstm


class BILSTM(object):
    """BiLSTM model"""

    def __init__(self, vocab, opt):

        self.pc = dy.ParameterCollection()
        # Hyper parameters
        self.embed_dim = opt.embed_dim
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.dropout
        # Word embeddings
        self.E = self.pc.add_lookup_parameters((len(vocab.id2w), self.embed_dim), name='E')
        # LSTMs
        self.fwd_lstm = lstm.LSTM(self.embed_dim, self.hidden_dim, self.pc, dropout=self.dropout)
        self.bwd_lstm = lstm.LSTM(self.embed_dim, self.hidden_dim, self.pc, dropout=self.dropout)
        # Load pretrained word embeddings
        if opt.pretrained_wembs is not None:
            print('Initializing word embeddings from %s' % opt.pretrained_wembs)
            sys.stdout.flush()
            word_vectors = np.load(opt.pretrained_wembs)
            self.E.init_from_array(word_vectors)

    def init(self, update=True, test=False):
        # Init LSTMs
        self.fwd_lstm.init(update=update, test=test)
        self.bwd_lstm.init(update=update, test=test)

    def prepare_batch(self, batch):
        bsize = len(batch)

        batch_len = max(len(s) for s in batch)

        x_fwd = np.zeros((batch_len, bsize), dtype=int)
        x_bwd = np.zeros((batch_len, bsize), dtype=int)
        masks = np.zeros((batch_len, bsize), dtype=float)

        for i in range(bsize):
            sent = batch[i][:]
            masks[:len(sent), i] = batch_len / len(sent)
            while len(sent) < batch_len:
                sent.append(0)
            x_fwd[:, i] = sent
            sent = batch[i][:]
            while len(sent) < batch_len:
                sent.insert(0, 0)
            x_bwd[:, i] = sent[::-1]

        return x_fwd, x_bwd, masks

    def compose_batch(self, x, a, update=True, test=False):
        x_fwd, x_bwd, masks = self.prepare_batch(x)
        _, bsize = x_fwd.shape
        # Init
        self.init(update=update, test=test)
        # Embed words
        word_vectors_fwd = [dy.lookup_batch(self.E, w, update=True) for w in x_fwd]
        word_vectors_bwd = [dy.lookup_batch(self.E, w, update=True) for w in x_bwd]
        # Initial hidden states
        h_0, c_0 = dy.zeros(self.hidden_dim, batch_size=bsize), dy.zeros(self.hidden_dim, batch_size=bsize)
        # Forward encodings
        fwd_h = []
        h, c = h_0, c_0
        for wv, mask in zip(word_vectors_fwd, masks):
            h, c = self.fwd_lstm(h, c, wv)
            masked_h = dy.cmult(h, dy.inputTensor(mask, batched=True))
            fwd_h.append(masked_h)
        # Backward encodings
        bwd_h = []
        h, c = h_0, c_0
        for wv, mask in zip(word_vectors_bwd, masks):
            h, c = self.bwd_lstm(h, c, wv)
            masked_h = dy.cmult(h, dy.inputTensor(mask, batched=True))
            bwd_h.append(masked_h)
        # Mean pooling
        fwd_h = dy.mean_dim(dy.concatenate_cols(fwd_h), d=[1], b=False)
        bwd_h = dy.mean_dim(dy.concatenate_cols(bwd_h), d=[1], b=False)
        # concatenate
        h = dy.concatenate([fwd_h, bwd_h])
        # Return representation
        return h

    def compare_batch(self, words, actions, update=True, test=False):
        """Run over batch"""

        x1 = [w[0] for w in words]
        x2 = [w[1] for w in words]

        h1 = self.compose_batch(x1, None, update=True, test=False)
        h2 = self.compose_batch(x2, None, update=True, test=False)

        delta = dy.concatenate([h1, h2, dy.cmult(h1, h2), dy.abs(h1 - h2)])

        return delta

    def compose(self, x, a, update=True, test=False, NT=False):
        """Compose a sentence into a vector"""
        return self.compose_batch([x], [a], update=update, test=test)

    def compare(self, x, y, ax, ay, update=True, test=False):
        """Compare sentence representations and return feature vector delta for the classifier"""
        hx = self.compose(x, ax, update, test)
        hy = self.compose(y, ay, update, test)
        delta = dy.concatenate([hx, hy, dy.cmult(hx, hy), dy.abs(hx - hy)])
        return delta



