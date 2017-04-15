#!/usr/bin/env python

from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
from codec import codecs_manager
from base import *
from util.param_util import convert_params


class RandomForestRegressor(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['random_state', 'n_estimators', 'max_depth', 'min_samples_split', 'max_leaf_nodes'],
            strs=['max_features'],
        )

        if 'max_depth' not in out_params:
            out_params.setdefault('max_leaf_nodes', 2000)

        # EAFP... convert max_features to int if it is a number.
        try:
            out_params['max_features'] = float(out_params['max_features'])
            max_features_int = int(out_params['max_features'])
            if out_params['max_features'] == max_features_int:
                out_params['max_features'] = max_features_int
        except:
            pass

        self.estimator = _RandomForestRegressor(**out_params)

    def summary(self):
        df = pd.DataFrame({'feature': self.columns,
                           'importance': self.estimator.feature_importances_.ravel()})
        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec('algos.RandomForestRegressor', 'RandomForestRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.forest', 'RandomForestRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'DecisionTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
