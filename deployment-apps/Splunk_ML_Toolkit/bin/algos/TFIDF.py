#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
from codec import codecs_manager
from base import *
from util.param_util import convert_params


class TFIDF(TransformerMixin):
    def __init__(self, options):
        if len(options.get('variables', [])) != 1 or len(options.get('explanatory_variables', [])) > 0:
            raise RuntimeError('Syntax error: You must specify exactly one field')

        self.variable = options['variables'][0]

        out_params = convert_params(
            options.get('params', {}),
            ints=['max_features'],
            floats=['max_df', 'min_df'],
            strs=[
                'ngram_range',
                'stop_words',
                'analyzer',
                'norm',
                'token_pattern'
            ],
            aliases={
            }
        )

        if 'ngram_range' in out_params.keys():
            # sklearn wants a tuple for ngram_range
            # SPL won't accept a tuple as an option's value
            # but users providing a range such as 1-10, e.g. kmeans k=2-10
            try:
                out_params['ngram_range'] = tuple(int(i) for i in out_params['ngram_range'].split('-'))
                assert len(out_params['ngram_range']) == 2
            except:
                raise RuntimeError('Syntax Error: ngram_range requires a range, e.g. ngram_range=1-5')


        # TODO: Maybe let the user know that we make this change.
        out_params.setdefault('max_features', 100)

        self.estimator = _TfidfVectorizer(**out_params)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.TFIDF', 'TFIDF', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.feature_extraction.text', 'TfidfVectorizer', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.feature_extraction.text', 'TfidfTransformer', SimpleObjectCodec)
        codecs_manager.add_codec('scipy.sparse.dia', 'dia_matrix', SimpleObjectCodec)
