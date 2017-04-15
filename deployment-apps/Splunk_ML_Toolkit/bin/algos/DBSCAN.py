#!/usr/bin/env python

from sklearn.cluster import DBSCAN as _DBSCAN
import pandas as pd
import numpy as np

from util.base_util import match_field_globs
from util.param_util import convert_params
from base import *


class DBSCAN(BaseMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(options.get('params', {}), floats=['eps'])

        self.estimator = _DBSCAN(**out_params)

    def fit_predict(self, X, options=None, output_name=None):
        if options is not None:
            self.variables = match_field_globs(X.columns, self.variables)
            self.drop_unused_fields(X, self.variables)
            self.drop_na_columns(X)
            self.warn_on_missing_fields(X, self.variables)
            self.filter_non_numeric(X)
            X = pd.get_dummies(X, prefix_sep='=', sparse=True)

            output_name = 'cluster'
            output = pd.DataFrame({output_name: np.empty(len(X))})
            output[output_name] = np.nan

            nans = self.drop_na_rows(X)
            y_hat = self.estimator.fit_predict(X.values)

            output.ix[~nans, output_name] = y_hat

            return output
