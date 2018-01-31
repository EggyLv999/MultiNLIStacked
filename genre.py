from __future__ import print_function, division

import sys

import dynet as dy
from dynn import layers
from dynn import activations

class GenreClassifier(object):

    def __init__(self, pc):
        self.pc = pc.add_subcollection('genre')

    def init(self, *args):
        pass

    def disc_loss(self, x, label):
        raise NotImplemented()

    def disc_loss_batch(self, x, labels):
        raise NotImplemented()

    def gen_loss(self, x):
        raise NotImplemented()

    def gen_loss_batch(self, x):
        raise NotImplemented()


class MLPGenreClassifier(GenreClassifier):

    def __init__(self, num_classes, num_layers, input_dim, hidden_dim, pc, activation=activations.relu, dropout=0.0, label_smoothing=0):
        super(MLPGenreClassifier, self).__init__(pc)
        # Hidden dim
        self.nc = num_classes
        self.nl = num_layers
        self.di = input_dim
        self.dh = hidden_dim
        self.drop = dropout
        # Layers
        self.hiddens = []
        for i in range(self.nl):
            self.hiddens.append(layers.DenseLayer(input_dim, hidden_dim, self.pc, activation=activation, dropout=dropout))
            input_dim=hidden_dim
        self.linear = layers.DenseLayer(input_dim, num_classes, self.pc, activation=activations.identity, dropout=dropout)

    def init(self, update=True, test=False):
        [hidden.init(update=update, test=test) for hidden in self.hiddens]
        self.linear.init(update=update, test=test)
        self.test = test

    def score(self, x):
        h = x
        for hidden in self.hiddens:
            h = hidden(h)
        self.h = h
        return self.linear(h)

    def disc_loss(self, x, label):
        return self.disc_loss_batch(x, [label])

    def disc_loss_batch(self, x, labels):
        x = dy.nobackprop(x)
        if isinstance(labels[0], int):
            return dy.mean_batches(dy.pickneglogsoftmax_batch(self.score(x), labels))
        else:
            p_gold = dy.inputTensor(labels.T, batched=True)
            return dy.mean_batches(-dy.dot_product(p_gold, dy.log_softmax(self.score(x))))

    def gen_loss(self, x):
        return self.disc_loss_batch(x, [label])

    def gen_loss_batch(self, x):
        self.init(update=False, test=self.test)
        return -dy.mean_batches(dy.mean_elems(dy.log_softmax(self.score(x))))

def get_genre_classifier(num_classes, pc, opt):
    if opt.genre_classifier == 'mlp':
        return MLPGenreClassifier(num_classes,
                             opt.genre_classifier_num_layers,
                             opt.classifier_hidden_dim,
                             opt.genre_classifier_hidden_dim,
                             pc, dropout=opt.dropout,
                             label_smoothing=opt.label_smoothing)
    else:
        print('Unknown genre classifier: %s' % opt.genre_classifier, file=sys.stderr)

