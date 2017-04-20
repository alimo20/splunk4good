#!/usr/bin/env python

from sklearn.cluster import KMeans as _KMeans
from codec import codecs_manager
from base import *
from itertools import izip
import numpy as np
import pandas as pd
from util.param_util import convert_params
import math


class KMeans(ClustererMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['k', 'random_state'],
            aliases={
                'k': 'n_clusters',
                'random_state': 'random_state'
            }
        )

        self.estimator = _KMeans(**out_params)

    def summary(self):
        df = pd.DataFrame(data=self.estimator.cluster_centers_, columns=self.columns)
        df['cluster'] = pd.Series(map(str, range(len(self.estimator.cluster_centers_))), df.index)
        idf = pd.DataFrame(data=[self.estimator.inertia_], columns=['inertia'])
        return pd.concat([df, idf], axis=0, ignore_index=True)

    def predict(self, X, options=None, output_name=None):
        if options is not None:
            X = self.preprocess_predict(X)
            df = X[self.columns].values

            # Allocate output DataFrame
            output_name = 'cluster'
            output = pd.DataFrame({output_name: np.empty(len(X))})
            output[output_name] = np.nan

            nans = self.drop_na_rows(X)
            y_hat = self.estimator.predict(X.values)

            output.ix[~nans, output_name] = y_hat
            self.rename_columns(output, options)

            cluster_ctrs = self.estimator.cluster_centers_
            cluster_pred = output['cluster'].values
            assert (len(df) == len(cluster_pred))
            dist = [float('nan') if math.isnan(cluster) else np.sum(np.square(cluster_ctrs[cluster] - row)) for
                    (cluster, row) in izip(cluster_pred, df)]
            output['cluster_distance'] = pd.Series(dist, output.index)
            output['cluster'] = ['' if math.isnan(v) else str('%.0f' % v) for v in output['cluster'].values]
            return output

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.KMeans', 'KMeans', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.cluster.k_means_', 'KMeans', SimpleObjectCodec)
