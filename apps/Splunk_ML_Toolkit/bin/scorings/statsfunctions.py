#!/usr/bin/env python

import pandas as pd

import cexc
from base_scoring import (
    BaseScoring,
    SingleArrayScoringMixin,
    DoubleArrayScoringMixin,
)
from util.param_util import convert_params
from util.scoring_util import (
    load_scoring_function,
    add_default_params,
    prepare_statistical_scoring_data,
    warn_order_not_preserved,
    check_limits_param,
    get_and_check_fields_two_1d_arrays,
    warn_on_same_fields,
    get_field_identifiers,
)

messages = cexc.get_messages_logger()


class DescribeScoring(SingleArrayScoringMixin, BaseScoring):
    """Implements scipy.stats.describe"""

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            ints=['ddof'],
            bools=['bias']
        )
        converted_params = add_default_params(converted_params, {'ddof': 1, 'bias': True})
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        _meta_params = {'field_identifiers': get_field_identifiers(prepared_df, "Field")}
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a scipy "describe" object. """
        # Create dictionary of results. Split minmax field for easier searching
        dict_results = {'kurtosis': result.kurtosis, 'mean': result.mean, 'nobs': result.nobs,
                        'skewness': result.skewness, 'variance': result.variance, 'min': result.minmax[0],
                        'max': result.minmax[1]}
        # Update dict_results with field identifiers
        dict_results.update(_meta_params['field_identifiers'])
        output_df = pd.DataFrame(dict_results)
        return output_df


class MomentScoring(SingleArrayScoringMixin, BaseScoring):
    """Implements scipy.stats.moment"""

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            ints=['moment']
        )
        converted_params = add_default_params(converted_params, {'moment': 1})
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        _meta_params = {'field_identifiers': get_field_identifiers(prepared_df, "Field")}
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a numpy array."""
        dict_results = {'moment': result}
        dict_results.update(_meta_params['field_identifiers'])
        output_df = pd.DataFrame(dict_results)
        return output_df


class TMeanScoring(SingleArrayScoringMixin, BaseScoring):
    """Implements scipy.stats.tmean"""

    @staticmethod
    def convert_param_types(params):
        # When user explicitly sets a limit to None, pop it since we expect floats and set default to None
        for p in ['upper_limit', 'lower_limit']:
            if params.get(p, '').lower() == 'none':
                params.pop(p)

        converted_params = convert_params(
            params,
            floats=['lower_limit', 'upper_limit']
        )
        converted_params = add_default_params(converted_params, {'lower_limit': None, 'upper_limit': None})
        converted_params = check_limits_param(converted_params)
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        _meta_params = {}
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a float."""
        output_df = pd.DataFrame(result, columns=['trimmed-mean'], index=[''])
        return output_df


class TVarScoring(SingleArrayScoringMixin, BaseScoring):
    """Implements scipy.stats.tvar """

    @staticmethod
    def convert_param_types(params):
        # When user explicitly sets a limit to None, pop it since we expect floats and set default to None
        for p in ['upper_limit', 'lower_limit']:
            if params.get(p, '').lower() == 'none':
                params.pop(p)

        converted_params = convert_params(
            params,
            ints=['ddof'],
            floats=['lower_limit', 'upper_limit']
        )
        converted_params = add_default_params(converted_params, {'lower_limit': None, 'upper_limit': None, 'ddof': 1})
        converted_params = check_limits_param(converted_params)
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        _meta_params = {}
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a float."""
        output_df = pd.DataFrame(result, columns=['trimmed-variance'], index=[''])
        return output_df


class TrimScoring(SingleArrayScoringMixin, BaseScoring):
    """Implements either scipy.stats.trim1 or scipy.stats.trimboth. """

    @staticmethod
    def load_scoring_function_with_options(options):
        """ Decide whether to load trim1 or trimboth based on options.

        When parameter 'tail' is 'both', load 'trimboth'. Otherwise,
        load 'trim1'.
        """
        tail = options.get('params', {}).get('tail', 'both')  # default is 'both'
        if tail not in ['left', 'right', 'both']:
            msg = 'Value error: parameter "tail" must be one of: "left", "right" or "both". Found tail="{}".'
            raise RuntimeError(msg.format(tail))

        scoring_function_name = 'trimboth' if tail == 'both' else 'trim1'
        scoring_function = load_scoring_function('scipy.stats', scoring_function_name)
        return scoring_function

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            strs=['tail'],
            floats=['proportiontocut']
        )
        converted_params = add_default_params(converted_params, {'tail': 'both', 'proportiontocut': 0})
        warn_order_not_preserved()
        if converted_params['tail'] == 'both':
            converted_params.pop('tail')  # Remove 'tail' parameter since scipy's 'trimboth' doesn't accept it.
        if not 0 <= converted_params['proportiontocut'] <= 1:
            msg = 'Value error: parameter "proportiontocut" must be between 0 and 1, but found proportiontocut="{}".'
            raise RuntimeError(msg.format(converted_params['proportiontocut']))
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        _meta_params = {'field_identifiers': ['trimmed({})'.format(i) for i in list(prepared_df.columns)]}
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a numpy array."""
        output_df = pd.DataFrame(result, columns=_meta_params['field_identifiers'])
        return output_df


class PearsonrScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.pearsonr """

    def handle_options(self, options):
        """ Pearsonr requires each array to be made of exactly 1 field. """
        params = options.get('params', {})
        params = self.convert_param_types(params)
        a_fields, b_fields = get_and_check_fields_two_1d_arrays(options, self.scoring_name)
        return params, a_fields, b_fields

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(params)
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        warn_on_same_fields(list(a_array.columns), list(b_array.columns))
        _meta_params = {}
        return a_array, b_array, _meta_params

    @staticmethod
    def create_output(scoring_name, result, _meta_params=None):
        """ Result is a tuple. """
        dict_results = {'r': result[0], 'two-tailed p-value': result[1]}
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class SpearmanrScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.spearmanr """

    def handle_options(self, options):
        """ Spearmanr requires each array to be made of exactly 1 field. """
        params = options.get('params', {})
        params = self.convert_param_types(params)
        a_fields, b_fields = get_and_check_fields_two_1d_arrays(options, self.scoring_name)
        return params, a_fields, b_fields

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(params)
        return converted_params

    @staticmethod
    def prepare_and_check_data(df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        warn_on_same_fields(list(a_array.columns), list(b_array.columns))
        _meta_params = {}
        return a_array, b_array, _meta_params

    @staticmethod
    def create_output(scoring_name, result, _meta_params=None):
        """ Result is a tuple. """
        dict_results = {'correlation': result[0], 'two-tailed p-value': result[1]}
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df
