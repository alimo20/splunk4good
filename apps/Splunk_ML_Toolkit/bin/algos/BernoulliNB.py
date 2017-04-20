#!/usr/bin/env python

from sklearn.naive_bayes import BernoulliNB as _BernoulliNB
from base import EstimatorMixin
from util.param_util import convert_params
from codec import codecs_manager
import numpy as np


class BernoulliNB(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(options.get('params', {}), floats=['alpha', 'binarize'], bools=['fit_prior'])

        self.estimator = _BernoulliNB(**out_params)
        self.is_classifier = True
        self.classes = None
        self.columns = None

    def partial_fit(self, X, handle_new_cat):
        X, y, columns = self.preprocess_fit(X)
        if self.classes is None:
            self.classes = np.unique(y)
            self.estimator.partial_fit(X.values, y.values, self.classes)
            self.columns = columns
        else:
            self.handle_categorical(X, y, handle_new_cat, self.columns, self.classes)
            if X.empty:
                return
            self.estimator.partial_fit(X.values, y.values)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.BernoulliNB', 'BernoulliNB', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.naive_bayes', 'BernoulliNB', SimpleObjectCodec)
