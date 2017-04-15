#!/usr/bin/env python

from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from codec import codecs_manager
from codec.codecs import NoopCodec

from base import EstimatorMixin
from util.param_util import convert_params


class SGDClassifier(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['fit_intercept'],
            ints=['random_state', 'n_iter'],
            floats=['l1_ratio', 'alpha', 'eta0', 'power_t'],
            strs=['loss', 'penalty', 'learning_rate'],
        )

        if 'loss' in out_params:
            try:
                assert (out_params['loss'] in ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])
            except AssertionError:
                raise RuntimeError(
                    'Value for parameter "loss" has to be one of "hinge", "log", "modified_huber", "squared_hinge", or "perceptron"')
        self.scaler = StandardScaler()
        self.estimator = _SGDClassifier(**out_params)
        self.columns = None
        self.classes = None
        self.is_classifier = True

    def fit(self, X):
        X, y, self.columns = self.preprocess_fit(X)
        scaled_X = self.scaler.fit_transform(X.values)
        self.estimator.fit(scaled_X, y.values)

    def partial_fit(self, X, handle_new_cat):
        X, y, columns = self.preprocess_fit(X)
        if self.classes is None:
            self.classes = np.unique(y)
            self.scaler.partial_fit(X.values)
            scaled_X = self.scaler.transform(X.values)
            self.estimator.partial_fit(scaled_X, y, classes=self.classes)
            self.columns = columns
        else:
            self.handle_categorical(X, y, handle_new_cat, self.columns, self.classes)
            if X.empty:
                return
            self.scaler.partial_fit(X.values)
            scaled_X = self.scaler.transform(X.values)
            self.estimator.partial_fit(scaled_X, y)

    def predict(self, X, options=None, output_name=None):
        X = self.preprocess_predict(X)

        # Allocate output DataFrame
        output_name = 'predicted(%s)' % self.response_variable
        output = pd.DataFrame({output_name: np.empty(len(X))})
        output[output_name] = np.nan

        nans = self.drop_na_rows(X)
        scaled_X = self.scaler.transform(X.values)
        y_hat = self.estimator.predict(scaled_X)

        output.ix[~nans, output_name] = y_hat
        self.rename_columns(output, options)

        return output

    def summary(self):
        df = pd.DataFrame()
        n_classes = len(self.estimator.classes_)
        limit = 1 if n_classes == 2 else n_classes

        for i, c in enumerate(self.estimator.classes_[:limit]):
            cdf = pd.DataFrame({'feature': self.columns,
                                'coefficient': self.estimator.coef_[i].ravel()})
            cdf = cdf.append(pd.DataFrame({'feature': ['_intercept'],
                                           'coefficient': [self.estimator.intercept_[i]]}))
            cdf['class'] = c
            df = df.append(cdf)

        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.SGDClassifier', 'SGDClassifier', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.linear_model.stochastic_gradient', 'SGDClassifier', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.preprocessing.data', 'StandardScaler', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.linear_model.sgd_fast', 'Hinge', NoopCodec)
        codecs_manager.add_codec('sklearn.linear_model.sgd_fast', 'Log', NoopCodec)
        codecs_manager.add_codec('sklearn.linear_model.sgd_fast', 'ModifiedHuber', NoopCodec)
        codecs_manager.add_codec('sklearn.linear_model.sgd_fast', 'SquaredHinge', NoopCodec)
