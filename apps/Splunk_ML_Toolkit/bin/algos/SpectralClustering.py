#!/usr/bin/env python
from sklearn.cluster import SpectralClustering as _SpectralClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from util.base_util import match_field_globs
from util.param_util import convert_params
from base import *


class SpectralClustering(BaseMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            floats=['gamma'],
            strs=['affinity'],
            ints=['k', 'random_state'],
            aliases={
                'k': 'n_clusters',
                'random_state': 'random_state'
            }
        )

        self.estimator = _SpectralClustering(**out_params)
        self.scaler = StandardScaler()

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

            if len(X) > 0 and len(X) <= self.estimator.n_clusters:
                raise RuntimeError("k must be smaller than the number of events used as input")

            scaled = self.scaler.fit_transform(X.values)
            y_hat = self.estimator.fit_predict(scaled)
            y_hat = ['' if np.isnan(v) else str('%.0f' % v) for v in y_hat]

            output.ix[~nans, output_name] = y_hat

            return output
