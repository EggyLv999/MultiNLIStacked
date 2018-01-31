from __future__ import print_function, division

import sys
import numpy as np
import dynet as dy

from dynn import lstm


class BaseModel(object):
    """Base class"""

    def init(self, update=True, test=False):
        pass

    def auxiliary_loss(self):
        """Any auxiliary loss term (regularizer, etc...)"""
        return dy.zeros(1)

    def compose_batch(self, x, a, update=True, test=False, NT=False):
        """Compose a batch of sentences into a vector"""
        pass

    def compose(self, x, a, update=True, test=False):
        """Compose a sentence into a vector"""
        return self.compose([x], [a], update=update, test=test)

    def compare_batch(self, words, actions, update=True, test=False):
        """Compare sentence representations and return feature vector delta for the classifier"""
        pass

    def compare(self, x, y, ax, ay, update=True, test=False):
        """Compare sentence representations and return feature vector delta for the classifier"""
        return self.compare_batch([[x,y]], None, update=update, test=test)


