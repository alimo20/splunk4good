#!/usr/bin/env python

from sklearn.cluster import Birch as _Birch
from base import *
from codec.flatten import flatten, expand
from codec.codecs import BaseCodec
from codec import codecs_manager
from util.param_util import convert_params


class BirchCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        """Birch has circular references and must be flattened."""
        flat_obj, refs = flatten(obj)

        return {
            '__mlspl_type': [type(obj).__module__, type(obj).__name__],
            'dict': flat_obj.__dict__,
            'refs': refs
        }

    @classmethod
    def decode(cls, obj):
        import sklearn.cluster

        m = sklearn.cluster.birch.Birch.__new__(sklearn.cluster.birch.Birch)
        m.__dict__ = obj['dict']

        return expand(m, obj['refs'])


class Birch(ClustererMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=['k'],
            aliases={
                'k': 'n_clusters'
            }
        )

        self.estimator = _Birch(**out_params)
        self.columns = None

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
            func = super(self.__class__, self).predict
            return self.apply_in_chunks(X, func, 10000, options, output_name)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('sklearn.cluster.birch', 'Birch', BirchCodec)
        codecs_manager.add_codec('codec.flatten', 'Ref', SimpleObjectCodec)
        codecs_manager.add_codec('algos.Birch', 'Birch', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.cluster.birch', '_CFNode', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.cluster.birch', '_CFSubcluster', SimpleObjectCodec)
