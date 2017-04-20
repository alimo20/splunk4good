#!/usr/bin/env python

from sklearn.kernel_ridge import KernelRidge as _KernelRidge
from base import EstimatorMixin
from util.param_util import convert_params
from codec import codecs_manager


class KernelRidge(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            floats=['gamma']
        )

        out_params['kernel'] = 'rbf'

        self.estimator = _KernelRidge(**out_params)

    def predict(self, X, options=None, output_name=None):
        if options is not None:
            func = super(self.__class__, self).predict
            return self.apply_in_chunks(X, func, 1000, options, output_name)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.KernelRidge', 'KernelRidge', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.kernel_ridge', 'KernelRidge', SimpleObjectCodec)
