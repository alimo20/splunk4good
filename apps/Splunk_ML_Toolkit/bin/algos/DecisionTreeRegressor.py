#!/usr/bin/env python

from sklearn.tree import DecisionTreeRegressor as _DecisionTreeRegressor
from codec import codecs_manager
from base import *
from util.param_util import convert_params


class DecisionTreeRegressor(TreeEstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['random_state', 'max_depth', 'min_samples_split', 'max_leaf_nodes'],
            strs=['splitter', 'max_features']
        )

        if 'max_depth' not in out_params:
            out_params.setdefault('max_leaf_nodes', 2000)

        # whitelist valid values for splitter, as error raised by sklearn for invalid values is uninformative
        if 'splitter' in out_params:
            try:
                assert (out_params['splitter'] in ['best', 'random'])
            except AssertionError:
                raise RuntimeError('Invalid value for option splitter: "%s"' % out_params['splitter'])

        # EAFP... convert max_features to int if it is a number.
        try:
            out_params['max_features'] = float(out_params['max_features'])
            max_features_int = int(out_params['max_features'])
            if out_params['max_features'] == max_features_int:
                out_params['max_features'] = max_features_int
        except:
            pass

        self.estimator = _DecisionTreeRegressor(**out_params)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec('algos.DecisionTreeRegressor', 'DecisionTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'DecisionTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)
