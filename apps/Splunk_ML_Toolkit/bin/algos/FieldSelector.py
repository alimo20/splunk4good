#!/usr/bin/env python
import copy

from sklearn.feature_selection import GenericUnivariateSelect, f_classif, f_regression
from codec import codecs_manager
from codec.codecs import BaseCodec
from base import *
from util.param_util import convert_params


class GenericUnivariateSelectCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        obj = copy.deepcopy(obj)
        obj.score_func = obj.score_func.__name__

        return {
            '__mlspl_type': [type(obj).__module__, type(obj).__name__],
            'dict': obj.__dict__,
        }

    @classmethod
    def decode(cls, obj):
        from sklearn.feature_selection import f_classif, f_regression, GenericUnivariateSelect

        new_obj = GenericUnivariateSelect.__new__(GenericUnivariateSelect)
        new_obj.__dict__ = obj['dict']

        if new_obj.score_func == 'f_classif':
            new_obj.score_func = f_classif
        elif new_obj.score_func == 'f_regression':
            new_obj.score_func = f_regression
        else:
            raise ValueError('Unsupported GenericUnivariateSelect.score_func "%s"' % new_obj.score_func)

        return new_obj


class FieldSelector(SelectorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            ints=[],
            floats=['param'],
            strs=['type', 'mode'],
            aliases={
                'k': 'param',
                'type': 'score_func'
            }
        )

        if 'score_func' not in out_params:
            out_params['score_func'] = f_classif
        else:
            if out_params['score_func'].lower() == 'categorical':
                out_params['score_func'] = f_classif
            elif out_params['score_func'].lower() in ['numerical', 'numeric']:
                out_params['score_func'] = f_regression
            else:
                raise RuntimeError('type can either be categorical or numeric.')

        if 'mode' in out_params:
            if out_params['mode'] not in ('k_best', 'fpr', 'fdr', 'fwe', 'percentile'):
                raise RuntimeError('mode can only be one of the following: fdr, fpr, fwe, k_best, and percentile')

        self.estimator = GenericUnivariateSelect(**out_params)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.FieldSelector', 'FieldSelector', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.feature_selection.univariate_selection', 'GenericUnivariateSelect',
                                 GenericUnivariateSelectCodec)
