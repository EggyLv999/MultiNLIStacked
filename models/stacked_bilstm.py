from __future__ import print_function, division

import sys
import numpy as np
import dynet as dy

from dynn import lstm

from comparators import comparators

def transduce(lstm, xs, h0, c0, masks=None):
    """Helper function for LSTM transduction"""
    h, c = h0, c0
    hs = []
    if masks is not None:
        for x, m in zip(xs, masks):
            h, c = lstm(h, c, x)
            hs.append(h)
            m_e = dy.inputTensor(m, batched=True)
            h, c = dy.cmult(h, m_e), dy.cmult(c, m_e)
    else:
        for x in xs:
            h, c = lstm(h, c, x)
            hs.append(h)
    return hs

class Stacked_BILSTM(object):
    """Shorcut stacked BiLSTM model https://arxiv.org/pdf/1708.02312.pdf"""

    def __init__(self, vocab, opt):
        self.pc = dy.ParameterCollection()
        # Hyper parameters
        self.embed_dim = opt.embed_dim
        self.hidden_dims = opt.hidden_dim
        self.num_layers = len(self.hidden_dims)
        self.dropout = opt.dropout_encoder
        # Word embeddings
        self.E = self.pc.add_lookup_parameters((len(vocab.id2w), self.embed_dim), name='E')
        # LSTMs
        input_size = self.embed_dim
        self.fwd_lstms = []
        self.bwd_lstms = []
        lstm_type = lstm.CompactLSTM if opt.compact_lstm else lstm.LSTM
        for hidden_dim in self.hidden_dims:
            self.fwd_lstms.append(lstm_type(input_size, hidden_dim, self.pc, dropout=self.dropout))
            self.bwd_lstms.append(lstm_type(input_size, hidden_dim, self.pc, dropout=self.dropout))
            input_size += 2 * hidden_dim
        # Load pretrained word embeddings
        if opt.pretrained_wembs is not None:
            print('Initializing word embeddings from %s' % opt.pretrained_wembs)
            sys.stdout.flush()
            word_vectors = np.load(opt.pretrained_wembs)
            self.E.init_from_array(word_vectors)
        self.comparator = comparators[opt.comparator]

    def init(self, update=True, test=False):
        # Init LSTMs
        [lstm.init(update=update, test=test) for lstm in self.fwd_lstms]
        [lstm.init(update=update, test=test) for lstm in self.bwd_lstms]

    def prepare_batch(self, batch):
        bsize = len(batch)

        batch_len = max(len(s) for s in batch)

        x = np.zeros((batch_len, bsize), dtype=int)
        add_masks = np.zeros((batch_len, bsize), dtype=float)
        mul_masks = np.zeros((batch_len, bsize), dtype=float)

        for i in range(bsize):
            sent = batch[i][:]
            # For max pooling we want additive masks with 0/-\infty
            add_masks[len(sent):, i] = -np.inf
            # For mean pooling we want 0/1masks (more or less, need to reweight for averaging)
            mul_masks[:len(sent), i] = 1 # batch_len / len(sent)
            # Fill sentence indices
            while len(sent) < batch_len:
                sent.append(0)
            x[:, i] = sent

        return x, add_masks, mul_masks


    def compute_hidden(self, xs, lstm_list, masks):
        """Compute hidden states with skip connections"""
        # TODO: Fix this to match the paper (interleaving bilstms)
        slen, bsize = masks.shape
        hs = []
        for i, lstm in enumerate(lstm_list):
            h0, c0 = dy.zeros(self.hidden_dims[i]), dy.zeros(self.hidden_dims[i])
            hs = transduce(lstm, xs, h0, c0)
            # concatenate output with hidden states of last one
            if i < len(lstm_list) - 1:
                xs = [dy.concatenate([x, h]) for x, h in zip(xs, hs)]
        # Masking (with negative infinity for max pooling)
        hs = [dy.inputTensor(mask, batched=True) + h for h, mask in zip(hs, masks)]
        # mean pooling
        #return dy.mean_dim(dy.concatenate_cols(hs), d=[1], b=False)
        # Max pooling: this is a hack to use the cudnn maxpooling until dynet's max_dim gets fatser
        # Reshape as (dh, L, 1) "image"
        h = dy.reshape(dy.concatenate_cols(hs), (self.hidden_dims[-1], slen, 1), batch_size=bsize)
        # 2D pooling with convenient kernel size
        max_pooled = dy.maxpooling2d(h, ksize=[1, slen], stride=[1,1])
        # Reshape as vector
        return dy.reshape(max_pooled, (self.hidden_dims[-1],), batch_size=bsize)

    def compute_hidden_interleaved(self, xs, add_masks, mul_masks):
        """Compute hidden states with skip connections"""
        slen, bsize = add_masks.shape
        fwd_hs, bwd_hs = [], []
        for i, (fwd_lstm, bwd_lstm) in enumerate(zip(self.fwd_lstms, self.bwd_lstms)):
            h0, c0 = dy.zeros(self.hidden_dims[i], batch_size=bsize), dy.zeros(self.hidden_dims[i], batch_size=bsize)
            # Run forward lstm
            fwd_hs = transduce(fwd_lstm, xs, h0, c0, mul_masks)
            # Run backward lstm
            bwd_hs = transduce(bwd_lstm, xs[::-1], h0, c0, mul_masks[::-1])
            # concatenate output with hidden states of last one
            if i < self.num_layers - 1:
                xs = [dy.concatenate([x, fh, bh]) for x, fh, bh in zip(xs, fwd_hs, bwd_hs[::-1])]
        # Masking (with negative infinity for max pooling)
        hs = [dy.inputTensor(mask, batched=True) + dy.concatenate([fh, bh]) for fh, bh, mask in zip(fwd_hs, bwd_hs[::-1], add_masks)]
        # Max pooling: this is a hack to use the cudnn maxpooling until dynet's max_dim gets fatser
        # Reshape as (dh, L, 1) "image"
        #h = dy.reshape(dy.concatenate_cols(hs), (2 * self.hidden_dims[-1], slen, 1), batch_size=bsize)
        h = dy.concatenate_cols(hs)
        self.h = h
        h = dy.reshape(h, (2 * self.hidden_dims[-1], slen, 1), batch_size=bsize)
        # 2D pooling with convenient kernel size
        max_pooled = dy.maxpooling2d(h, ksize=[1, slen], stride=[1,1])
        # Reshape as vector
        return dy.reshape(max_pooled, (2 * self.hidden_dims[-1],), batch_size=bsize)

    def compose_batch(self, x, a, update=True, test=False, NT=False):
        """Compose a sentence into a vector"""
        # Create mask and all
        x, add_masks, mul_masks = self.prepare_batch(x)
        # Initialize parameters
        self.init(update=update, test=test)
        # Embed words
        input_words = [dy.lookup_batch(self.E, w) for w in x]
        self.input_words = input_words
        # Compute hidden state according to Nie et al 2017
        output = self.compute_hidden_interleaved(input_words, add_masks, mul_masks)
        return output
        # compute hidden state
        #output_fwd = self.compute_hidden(input_fwd, self.fwd_lstms, masks)
        #output_bwd = self.compute_hidden(input_bwd, self.bwd_lstms, masks)

        #return dy.concatenate([output_fwd, output_bwd])

    def compose(self, x, a, update=True, test=False):
        return self.compose([x], [a], update=update, test=test)

    def compare_batch(self, words, actions, update=True, test=False):
        """Run over batch"""
        batch_size = len(words)

        x = [w[0] for w in words]
        y = [w[1] for w in words]
        #hx = self.compose_batch(x, None, update=update, test=test)
        #hy = self.compose_batch(y, None, update=update, test=test)
        # Faster but takes more memory: encode all sentences (premise and hypothesis) as one batch
        h = self.compose_batch(x+y, None, update=update, test=test)
        # Split to get the premise/hypothesis encodings
        hx = dy.pick_batch_elems(h, range(batch_size))
        hy = dy.pick_batch_elems(h, range(batch_size, 2*batch_size))
        # extract features
        return self.comparator(hx, hy)

    def compare(self, x, y, ax, ay, update=True, test=False):
        """Compare sentence representations and return feature vector delta for the classifier"""
        return self.compare_batch([[x,y]], None, update=update, test=test)


