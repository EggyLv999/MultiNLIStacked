from __future__ import print_function, division

import dynet as dy
import numpy as np
from dynn import layers
from dynn import activations


class Classifier(object):

    def __init__(self, pc):
        self.pc = pc.add_subcollection('class')

    def init(self, *args):
        pass

    def loss(self, x, label):
        raise NotImplemented()

    def loss_batch(self, x, labels):
        raise NotImplemented()


class LinearClassifier(Classifier):

    def __init__(self, num_classes, input_dim, pc, dropout=0.0, label_smoothing=0, layer_norm=False):
        super(LinearClassifier, self).__init__(pc)

        self.linear = layers.DenseLayer(input_dim, num_classes, pc, activation=lambda x:x, dropout=dropout)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = layers.LayerNormalization(input_dim, self.pc)


    def init(self, update=True, test=False):
        self.linear.init(update=update, test=test)
        if self.layer_norm:
            self.norm.init()

    def score(self, x):
        if self.layer_norm:
            x = self.norm(x)
        return self.linear(x)

    def loss(self, x, label):
        return self.loss_batch(x, [label])

    def loss_batch(self, x, labels):
        if isinstance(labels[0], int):
            return dy.mean_batches(dy.pickneglogsoftmax_batch(self.score(x), labels))
        else:
            p_gold = dy.inputTensor(labels.T, batched=True)
            return dy.mean_batches(-dy.dot_product(p_gold, dy.log_softmax(self.score(x))))

class MLPClassifier(Classifier):

    def __init__(self, num_classes, num_layers, input_dim, hidden_dim, pc, activation=activations.relu, dropout=0.0, label_smoothing=0, residual=False):
        super(MLPClassifier, self).__init__(pc)
        # Hidden dim
        self.nc = num_classes
        self.nl = num_layers
        self.di = input_dim
        self.dh = hidden_dim
        self.drop = dropout
        self.residual = residual
        # Layers
        self.hiddens = []
        for i in range(self.nl):
            self.hiddens.append(layers.DenseLayer(input_dim, hidden_dim, pc, activation=activation, dropout=dropout))
            input_dim=hidden_dim
        self.linear = layers.DenseLayer(input_dim, num_classes, pc, activation=activations.identity, dropout=dropout)

    def init(self, update=True, test=False):
        [hidden.init(update=update, test=test) for hidden in self.hiddens]
        self.linear.init(update=update, test=test)

    def score(self, x):
        h = x
        for l, hidden in enumerate(self.hiddens):
            if self.residual and l>0:
                h = hidden(h) + h
            else:
                h = hidden(h)
        self.h = h
        return self.linear(h)

    def loss(self, x, label):
        return self.loss_batch(x, [label])

    def loss_batch(self, x, labels):
        if isinstance(labels[0], int):
            return dy.mean_batches(dy.pickneglogsoftmax_batch(self.score(x), labels))
        else:
            p_gold = dy.inputTensor(labels.T, batched=True)
            return dy.mean_batches(-dy.dot_product(p_gold, dy.log_softmax(self.score(x))))

class NeutralMLPClassifier(Classifier):

    def __init__(self, num_classes, num_layers, input_dim, hidden_dim, pc, activation=activations.relu, dropout=0.0,
                 label_smoothing=0, residual=False):
        super(NeutralMLPClassifier, self).__init__(pc)
        # Hidden dim
        self.nc = num_classes
        self.nl = num_layers
        self.di = input_dim
        self.dh = hidden_dim
        self.drop = dropout
        self.residual = residual
        # Layers
        self.hiddens = []
        for i in range(self.nl):
            self.hiddens.append(
                layers.DenseLayer(input_dim, hidden_dim, pc, activation=activation, dropout=dropout))
            input_dim = hidden_dim
        self.linear_n = layers.DenseLayer(input_dim, 2, pc, activation=activations.identity,
                                        dropout=dropout)
        self.linear_e_c = layers.DenseLayer(input_dim, 2, pc, activation=activations.identity,
                                        dropout=dropout)

    def init(self, update=True, test=False):
        [hidden.init(update=update, test=test) for hidden in self.hiddens]
        self.linear_n.init(update=update, test=test)
        self.linear_e_c.init(update=update, test=test)

    def score(self, x):
        h = x
        for l, hidden in enumerate(self.hiddens):
            if self.residual and l>0:
                h = hidden(h) + h
            else:
                h = hidden(h)
        self.h = h
        n_score = self.linear_n(h)
        e_c_score = self.linear_e_c(h)
        lp_neutral = dy.log_softmax(n_score)
        lp_e_c = dy.log_softmax(e_c_score)
        log_prob = dy.concatenate([lp_neutral[0], lp_neutral[1] + lp_e_c])
        return log_prob

    def loss(self, x, label):
        return self.loss_batch(x, [label])

    def loss_batch(self, x, labels):
        if isinstance(labels[0], int):
            return dy.mean_batches(-dy.pick_batch(self.score(x), labels))
        else:
            p_gold = dy.inputTensor(labels.T, batched=True)
            return dy.mean_batches(-dy.dot_product(p_gold, self.score(x)))

class GenClassifier(Classifier):

    def __init__(self, num_classes, num_layers, input_dim, hidden_dim, pc, activation=activations.relu, dropout=0.0, label_smoothing=0, residual=False):
        super(GenClassifier, self).__init__(pc)
        # Hidden dim
        self.nc = num_classes
        self.nl = num_layers
        self.di = input_dim
        self.dh = hidden_dim
        self.drop = dropout
        self.residual = residual
        # Layers
        self.hiddens = [[] for c in range(self.nc)]
        for i in range(self.nl):
            for c in range(self.nc):
                self.hiddens[c].append(layers.DenseLayer(input_dim, hidden_dim, pc, activation=activation, dropout=dropout))
                input_dim=hidden_dim
        self.linear = [layers.DenseLayer(input_dim, hidden_dim, pc, activation=activations.identity, dropout=dropout) for c in range(self.nc)]

    def init(self, update=True, test=False):
        [[hidden.init(update=update, test=test) for hidden in self.hiddens[c]] for c in range(self.nc)]
        [self.linear[c].init(update=update, test=test) for c in range(self.nc)]

    def score(self, x):
        h = x[0]
        hc = []
        for c in range(self.nc):
            for l, hidden in enumerate(self.hiddens[c]):
                if self.residual and l>0:
                    h = hidden(h) + h
                else:
                    h = hidden(h)
            hc.append(-dy.squared_distance(self.linear[c](h), x[1]))
        return dy.concatenate(hc)

    def loss(self, x, label):
        return self.loss_batch(x, [label])

    def loss_batch(self, x, labels):
        if isinstance(labels[0], int):
            return dy.mean_batches(dy.pickneglogsoftmax_batch(self.score(x), labels))
        else:
            p_gold = dy.inputTensor(labels.T, batched=True)
            return dy.mean_batches(-dy.dot_product(p_gold, dy.log_softmax(self.score(x))))

class BilinearClassifier(Classifier):
    
    def __init__(self, num_classes, input_dim, pc, dropout=0.0, label_smoothing=0):
        super(BilinearClassifier, self).__init__(pc)
        # Hidden dim
        self.nc = num_classes
        self.di = input_dim
        self.drop = dropout
        # Layers
        self.bilinear = []
        self.linear_x = []
        self.linear_y = []
        for i in range(self.nc):
            self.bilinear.append(self.pc.add_parameters((self.di, self.di), init=dy.UniformInitializer(np.sqrt(3)/self.di), name='bilinear-%d' % i))
            self.linear_x.append(self.pc.add_parameters(self.di, name='linear-x-i%d' % i))
            self.linear_y.append(self.pc.add_parameters(self.di, name='linear-y-i%d' % i))
        self.b = self.pc.add_parameters(self.nc, dy.ConstInitializer(0))

    def init(self, update=True, test=False):
        self.update, self.test = update, test

    def score(self, x):
        hx = x[:self.di]
        hy = x[self.di:2*self.di]
        scores = []
        for b, lx, ly  in zip(self.bilinear, self.linear_x, self.linear_y):
            s = dy.dot_product(hx, b.expr(self.update) * hy)
            s += dy.dot_product(lx.expr(self.update), hx)
            s += dy.dot_product(ly.expr(self.update), hy)
            scores.append(s)
        score = dy.concatenate(scores) + self.b.expr(self.update)
        return score

    def loss(self, x, label):
        return self.loss_batch(x, [label])

    def loss_batch(self, x, labels):
        if isinstance(labels[0], int):
            return dy.mean_batches(dy.pickneglogsoftmax_batch(self.score(x), labels))
        else:
            p_gold = dy.inputTensor(labels.T, batched=True)
            return dy.mean_batches(-dy.dot_product(p_gold, dy.log_softmax(self.score(x))))


def get_classifier(num_classes, pc, opt):
    if opt.classifier == 'mlp':
        return MLPClassifier(num_classes,
                             opt.classifier_num_layers,
                             opt.classifier_input_dim,
                             opt.classifier_hidden_dim,
                             pc, dropout=opt.dropout,
                             label_smoothing=opt.label_smoothing,
                             residual=opt.classifier_residual)
    if opt.classifier == 'gen':
        return GenClassifier(num_classes,
                             opt.classifier_num_layers,
                             opt.classifier_input_dim,
                             opt.classifier_hidden_dim,
                             pc, dropout=opt.dropout,
                             label_smoothing=opt.label_smoothing,
                             residual=opt.classifier_residual)
    elif opt.classifier == 'neutral_mlp':
        return NeutralMLPClassifier(num_classes,
                             opt.classifier_num_layers,
                             opt.classifier_input_dim,
                             opt.classifier_hidden_dim,
                             pc, dropout=opt.dropout,
                             label_smoothing=opt.label_smoothing,
                             residual=opt.classifier_residual)
    elif opt.classifier == 'bilinear':
        return BilinearClassifier(num_classes,
                                     2 * opt.hidden_dim[-1],
                                     pc)
    elif opt.classifier == 'linear':
        return LinearClassifier(num_classes, opt.classifier_input_dim, pc, dropout=opt.dropout,
                                label_smoothing=opt.label_smoothing, layer_norm = opt.classifier_layer_norm)

