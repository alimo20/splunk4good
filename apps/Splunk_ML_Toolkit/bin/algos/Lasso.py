#!/usr/bin/env python

from sklearn.linear_model import Lasso  as _Lasso
from codec import codecs_manager
from base import *
import pandas as pd
from util.param_util import convert_params


class Lasso(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(options.get('params', {}), floats=['alpha'])

        self.estimator = _Lasso(**out_params)

    def summary(self):
        df = pd.DataFrame({'feature': self.columns,
                           'coefficient': self.estimator.coef_.ravel()})
        idf = pd.DataFrame({'feature': ['_intercept'],
                            'coefficient': [self.estimator.intercept_]})
        return pd.concat([df, idf])

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.Lasso', 'Lasso', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.linear_model.coordinate_descent', 'Lasso', SimpleObjectCodec)
