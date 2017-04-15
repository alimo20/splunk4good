#!/usr/bin/env python

from sklearn.decomposition import PCA as _PCA
from base import *
from codec import codecs_manager
import cexc
from util.base_util import match_field_globs
from util.param_util import convert_params


class PCA(BaseMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['k'],
            aliases={
                'k': 'n_components'
            }
        )
        self.estimator = _PCA(**out_params)

    def fit(self, X):
        self.variables = match_field_globs(X.columns, self.variables)
        self.drop_unused_fields(X, self.variables)
        self.drop_na_columns(X)
        self.drop_na_rows(X)
        self.warn_on_missing_fields(X, self.variables)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.assert_any_fields(X)
        self.assert_any_rows(X)
        self.columns = self.sort_fields(X)

        self.estimator.fit(X.values)

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
            y_hat = y_hat[:, ~np.isnan(y_hat[0, :])]
            width = y_hat.shape[1]
            if width < self.estimator.n_components:
                cexc.messages.info('%s extracted maximum number of features (%d), '
                                   'which is less than the specified k (%d).'
                                   % (type(self).__name__, width, self.estimator.n_components))
            # Allocate output DataFrame
            columns = ['PC_%d' % (i + 1) for i in range(width)]

            output = pd.DataFrame(np.empty((length, width)), columns=columns)
            output.ix[:, columns] = np.nan
            output.ix[~nans, columns] = y_hat

            return output

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.PCA', 'PCA', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.decomposition.pca', 'PCA', SimpleObjectCodec)
