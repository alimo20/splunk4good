#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler as _StandardScaler
from base import *
from codec import codecs_manager
import pandas as pd
from util.base_util import match_field_globs
from util.param_util import convert_params


class StandardScaler(BaseMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['with_mean', 'with_std']
        )
        self.estimator = _StandardScaler(**out_params)
        self.columns = None

    def preprocess_fit(self, X):
        self.variables = match_field_globs(X.columns, self.variables)
        self.drop_unused_fields(X, self.variables)
        self.drop_na_columns(X)
        self.drop_na_rows(X)
        self.warn_on_missing_fields(X, self.variables)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.assert_any_fields(X)
        self.assert_any_rows(X)
        columns = self.sort_fields(X)
        return (X, columns)

    def fit(self, X):
        X, self.columns = self.preprocess_fit(X)
        self.estimator.fit(X)

    def partial_fit(self, X, handle_new_cat):
        X, columns = self.preprocess_fit(X)
        if self.columns is not None:
            self.handle_categorical(X, None, handle_new_cat, self.columns)
            if X.empty:
                return
        else:
            self.columns = columns
        self.estimator.partial_fit(X)

    def predict(self, X, options=None, output_name=None):
        if options is not None:
            self.drop_unused_fields(X, self.variables)
            self.drop_na_columns(X)
            self.warn_on_missing_fields(X, self.variables)
            self.filter_non_numeric(X)
            X = pd.get_dummies(X, prefix_sep='=', sparse=True)
            self.drop_unused_fields(X, self.columns)
            self.assert_any_fields(X)
            self.fill_missing_fields(X, self.columns)
            self.sort_fields(X)
            assert set(X.columns) == set(self.columns), 'Internal error: column mismatch'

            length = len(X)
            nans = self.drop_na_rows(X)

            y_hat = self.estimator.transform(X.values)

            # Allocate output DataFrame
            width = y_hat.shape[1]
            columns = ['SS_%s' % col for col in self.columns]

            output = pd.DataFrame(np.empty((length, width)), columns=columns)
            output.ix[:, columns] = np.nan
            output.ix[~nans, columns] = y_hat

            return output

    def summary(self):
        return pd.DataFrame({'fields': self.columns,
                             'mean': self.estimator.mean_,
                             'std': self.estimator.std_})

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.StandardScaler', 'StandardScaler', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.preprocessing.data', 'StandardScaler', SimpleObjectCodec)
