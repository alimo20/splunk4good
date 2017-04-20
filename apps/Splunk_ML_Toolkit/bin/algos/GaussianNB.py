#!/usr/bin/env python

from sklearn.naive_bayes import GaussianNB as _GaussianNB
import numpy as np

from base import EstimatorMixin
from util.param_util import convert_params
from codec import codecs_manager


class GaussianNB(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(options.get('params', {}))

        self.estimator = _GaussianNB()
        self.is_classifier = True
        self.classes = None
        self.columns = None

    def partial_fit(self, X, handle_new_cat):
        X, y, columns = self.preprocess_fit(X)
        if self.classes is None:
            self.classes = np.unique(y)
            self.estimator.partial_fit(X, y, classes=self.classes)
            self.columns = columns
        else:
            self.handle_categorical(X, y, handle_new_cat, self.columns, self.classes)
            if X.empty:
                return
            self.estimator.partial_fit(X, y)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.GaussianNB', 'GaussianNB', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.naive_bayes', 'GaussianNB', SimpleObjectCodec)
