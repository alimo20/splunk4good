#!/usr/bin/env python

from sklearn.linear_model import SGDRegressor as _SGDRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import cexc

from codec import codecs_manager
from base import EstimatorMixin
from util.param_util import convert_params


class SGDRegressor(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['fit_intercept'],
            ints=['random_state', 'n_iter'],
            floats=['l1_ratio', 'alpha', 'eta0', 'power_t'],
            strs=['penalty', 'learning_rate'],
        )

        self.scaler = StandardScaler()
        self.estimator = _SGDRegressor(**out_params)
        self.columns = None

    def fit(self, X):
        X, y, self.columns = self.preprocess_fit(X)
        scaled_X = self.scaler.fit_transform(X.values)
        self.estimator.fit(scaled_X, y.values)

    def partial_fit(self, X, handle_new_cat):
        X, y, columns = self.preprocess_fit(X)
        if self.columns is not None:
            self.handle_categorical(X, y, handle_new_cat, self.columns)
            if X.empty:
                return
        else:
            self.columns = columns
        self.scaler.partial_fit(X.values)
        scaled_X = self.scaler.transform(X.values)
        self.estimator.partial_fit(scaled_X, y.values)
        cexc.messages.warn('n_iter is set to 1 when partial fit is performed')

    def predict(self, X, options=None, output_name=None):

        if options is not None:
            X = self.preprocess_predict(X)

            # Allocate output DataFrame
            if output_name is None:
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
        df = pd.DataFrame({'feature': self.columns,
                           'coefficient': self.estimator.coef_.ravel()})
        idf = pd.DataFrame({'feature': '_intercept',
                            'coefficient': self.estimator.intercept_})
        return pd.concat([df, idf])

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.SGDRegressor', 'SGDRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.linear_model.stochastic_gradient', 'SGDRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.preprocessing.data', 'StandardScaler', SimpleObjectCodec)
