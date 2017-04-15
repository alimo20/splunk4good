#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from codec import codecs_manager
from base import *
from util.param_util import convert_params


class RandomForestClassifier(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['random_state', 'n_estimators', 'max_depth', 'min_samples_split', 'max_leaf_nodes'],
            strs=['max_features', 'criterion'],
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

        self.estimator = _RandomForestClassifier(class_weight='auto', **out_params)
        self.is_classifier = True

    def summary(self):
        df = pd.DataFrame({'feature': self.columns,
                           'importance': self.estimator.feature_importances_.ravel()})
        return df

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec('algos.RandomForestClassifier', 'RandomForestClassifier', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.forest', 'RandomForestClassifier', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'DecisionTreeClassifier', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
