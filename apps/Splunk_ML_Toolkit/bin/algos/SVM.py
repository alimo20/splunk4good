#!/usr/bin/env python

from sklearn.svm import SVC
from codec import codecs_manager
from base import *
from util.param_util import convert_params


class SVM(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(options.get('params', {}), floats=['gamma', 'C'])

        self.estimator = SVC(class_weight='auto', **out_params)
        self.is_classifier = True

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.SVM', 'SVM', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.svm.classes', 'SVC', SimpleObjectCodec)
