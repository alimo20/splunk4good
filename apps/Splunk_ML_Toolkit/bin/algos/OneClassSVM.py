#!/usr/bin/env python

from sklearn.svm.classes import OneClassSVM as _OneClassSVM
from codec import codecs_manager
from base import *
import pandas as pd
from util.param_util import convert_params


class OneClassSVM(ClustererMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            floats=['gamma', 'coef0', 'tol', 'nu'],
            ints=['degree'],
            bools=['shrinking'],
            strs=['kernel']
        )

        self.estimator = _OneClassSVM(**out_params)

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

            # Allocate output DataFrame
            output_name = 'isNormal'
            output = pd.DataFrame({output_name: np.empty(len(X))})
            output[output_name] = np.nan

            nans = self.drop_na_rows(X)
            y_hat = self.estimator.predict(X.values)

            output.ix[~nans, output_name] = y_hat
            self.rename_columns(output, options)
            return output

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.OneClassSVM', 'OneClassSVM', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.svm.classes', 'OneClassSVM', SimpleObjectCodec)

        # Not providing summary support at the moment.

        # def summary(self):
        #     df = pd.DataFrame()
        #     for i in range(len(self.estimator.support_)):
        #         for j in range(len(self.columns)):
        #             cdf = pd.DataFrame({'support_': self.estimator.support_[i], 'feature': self.columns[j],
        #                                 'support_vectors_': self.estimator.support_vectors_[i, j].ravel()})
        #             df = df.append(cdf)
        #     return df
