#!/usr/bin/env python
from collections import OrderedDict

import numpy as np
import pandas as pd

import cexc
from base_scoring import (
    BaseScoring,
    SingleArrayScoringMixin,
    DoubleArrayScoringMixin,
    TSAStatsToolsMixin,
)
from util.param_util import convert_params
from util.scoring_util import (
    add_default_params,
    prepare_statistical_scoring_data,
    get_field_identifiers,
    get_and_check_fields_two_1d_arrays,
    warn_on_num_samples,
    check_zero_method_param,
    check_alternative_param,
    warn_on_same_fields,
    validate_param_from_str_list,
    update_with_hypothesis_decision,
    check_alpha_param,
)

messages = cexc.get_messages_logger()


class NormalTestScoring(SingleArrayScoringMixin, BaseScoring):
    """Implements scipy.stats.normaltest """

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            floats=['alpha'],
        )
        converted_params = add_default_params(converted_params, {'alpha': 0.05})
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        # pop 'alpha' from params into _meta_params and add the field identifier
        _meta_params = {
            'field_identifiers': get_field_identifiers(prepared_df, "Field"),
            'alpha': self.params.pop('alpha'),
        }
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """Result is a scipy 'normaltest' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Update dict_results with field identifiers
        dict_results.update(_meta_params['field_identifiers'])
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the sample comes from a normal distribution.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results)
        return output_df


class TTestOneSampleScoring(SingleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.ttest_1samp """

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            floats=['popmean', 'alpha']
        )
        if 'popmean' not in converted_params:
            raise RuntimeError('Value error: float parameter "popmean" must be specified')

        converted_params = add_default_params(converted_params, {'alpha': 0.05})
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, fields, params):
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        # pop 'alpha' from params into _meta_params and add the field identifier
        _meta_params = {
            'field_identifiers': get_field_identifiers(prepared_df, "Field"),
            'alpha': self.params.pop('alpha'),
        }
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a scipy 'ttest_1samp' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Update dict_results with field identifiers
        dict_results.update(_meta_params['field_identifiers'])
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = ('the mean of the samples of independent observations is equal '
                           'to the population mean {}.'.format(self.params['popmean']))
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results)
        return output_df


class FOnewayScoring(SingleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.f_oneway """

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            floats=['alpha']
        )
        converted_params = add_default_params(converted_params, {'alpha': 0.05})
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, fields, params):
        if len(fields) < 2:
            raise RuntimeError('Value error: need 2 or more unique fields to perform a one-way ANOVA. '
                               'Please specify fields as: ..| score f_oneway <field_1> <field_2> ... <field_n>.')
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        if len(prepared_df) < 2:
            raise RuntimeError('Value error: need at least 2 samples to score on.')
        return prepared_df, _meta_params

    def score(self, df, options):
        """ f_oneway requires a sequence of 1d array inputs. """
        # Get preprocessed df and and meta-params for creating output
        preprocessed_df, _meta_params = self.prepare_and_check_data(df, self.variables, self.params)
        result = self.scoring_function(*preprocessed_df.values.T)  # No arguments taken
        # Create output with meta-params
        df_output = self.create_output(self.scoring_name, result, _meta_params)
        return df_output

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a scipy 'f_oneway' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the provided groups have the same population mean.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class KSTestScoring(SingleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.kstest """

    @staticmethod
    def convert_param_types(params):
        if 'cdf' in params:
            params['cdf'] = cdf_str = params['cdf'].lower()  # Convert to lowercase
        else:
            raise RuntimeError('Value error: the cumulative distribution function (cdf) must be specified. '
                               'Currently we support cdf values of: norm, lognorm and chi2.')

        # Currently we only support chi2, lognorm and norm cdfs.
        # Order is important since args is un-named sequence
        cdf_params_type_map = {
            'chi2': OrderedDict(
                [('df', 'int'),
                 ('loc', 'float'),
                 ('scale', 'float')]),
            'lognorm': OrderedDict(
                [('s', 'float'),
                 ('loc', 'float'),
                 ('scale', 'float')]),
            'norm': OrderedDict(
                [('loc', 'float'),
                 ('scale', 'float')])
        }

        if cdf_str not in cdf_params_type_map:
            msg = 'Value error: cdf="{}" is not currently supported. Please choose a cdf in: norm, lognorm or chi2.'
            raise RuntimeError(msg.format(cdf_str))

        # Get the parameters for the specified cdf. Assert that all are specified (since args is un-named sequence)
        unspecified_params = [p for p in cdf_params_type_map[cdf_str] if p not in [i.lower() for i in params]]
        if len(unspecified_params) > 0:
            raise RuntimeError('Value error: all distribution parameters must be given for cdf="{}". Please specify '
                               'the following parameters: {}.'.format(cdf_str, ', '.join(unspecified_params)))

        # Finally check that the correct param types are provided
        converted_params = convert_params(
            params,
            strs=['cdf', 'mode', 'alternative'],
            ints=[p for (p, p_type) in cdf_params_type_map[cdf_str].iteritems() if p_type == 'int'],
            floats=[p for (p, p_type) in cdf_params_type_map[cdf_str].iteritems() if p_type == 'float'] + ['alpha'],
        )
        # Check param values; scale, df and s must all be greater than zero
        for p in ['scale', 'df', 's']:
            if p in converted_params and converted_params[p] <= 0:
                raise RuntimeError('Value error: "{}" parameter must be greater than zero.'.format(p))

        # pop the cdf args from params since they are passed to sklearn as un-named sequence "args"
        converted_params['args'] = [converted_params.pop(k) for k in cdf_params_type_map[cdf_str]]
        # Add default params
        converted_params = add_default_params(
            converted_params, {'mode': 'approx', 'alternative': 'two-sided', 'alpha': 0.05}
        )
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, fields, params):
        if len(fields) > 1:  # ks_test takes a single 1d vector as input
            msg = ('Value error: scoring method "ks_test" requires a single field as input specified as '
                   '".. | score kstest <field> [options] ". Found {} passed fields.')
            raise RuntimeError(msg.format(len(fields)))
        prepared_df, _ = prepare_statistical_scoring_data(df, fields)
        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        return prepared_df, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is scipy Kstest object"""
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the two distributions are identical.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class MannWhitneyUScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.mannwhitneyu """

    def handle_options(self, options):
        """ Mannwhitneyu requires each array to be made of exactly 1 field. """
        params = options.get('params', {})
        params = self.convert_param_types(params)
        a_fields, b_fields = get_and_check_fields_two_1d_arrays(options, self.scoring_name)
        return params, a_fields, b_fields

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            strs=['alternative'],
            bools=['use_continuity'],
            floats=['alpha'],
        )
        converted_params = add_default_params(
            converted_params, {'alternative': 'two-sided', 'use_continuity': True, 'alpha': 0.05}
        )
        converted_params = check_alternative_param(converted_params)
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        # Warn if the same field is being compared and if number of samples is low
        warn_on_same_fields(list(a_array.columns), list(b_array.columns))
        warn_on_num_samples(a_array, 20)  # normal approximation
        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        return a_array, b_array, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a scipy 'Mannwhitneyu' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = ('it is equally likely that a randomly selected value from the first sample will be less '
                           'than or greater than a randomly selected value from the second sample')
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class WilcoxonScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.wilcoxon """

    def handle_options(self, options):
        """ Wilcoxon requires each array to be made of exactly 1 field. """
        params = options.get('params', {})
        params = self.convert_param_types(params)
        a_fields, b_fields = get_and_check_fields_two_1d_arrays(options, self.scoring_name)
        return params, a_fields, b_fields

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            strs=['zero_method'],
            bools=['correction'],
            floats=['alpha'],
        )
        converted_params = add_default_params(
            converted_params, {'zero_method': 'wilcox', 'correction': False, 'alpha': 0.05}
        )
        converted_params = check_zero_method_param(converted_params)
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        # Warn if the same field is being compared and if number of samples is low
        warn_on_same_fields(list(a_array.columns), list(b_array.columns))
        warn_on_num_samples(a_array, 20)  # normal approximation
        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        return a_array, b_array, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is scipy.stats 'Wilcoxon' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the related paired samples come from the same distribution.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class TTestTwoIndSampleScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.ttest_ind"""

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            bools=['equal_var'],
            floats=['alpha'],
        )
        converted_params = add_default_params(converted_params, {'equal_var': True, 'alpha': 0.05})
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        # pop 'alpha' from params into _meta_params and add the field identifier
        _meta_params = {
            'field_identifiers': get_field_identifiers(a_array, "A_field", b_array, "B_field"),
            'alpha': self.params.pop('alpha'),
        }
        return a_array, b_array, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a scipy 'Ttest_ind' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Update dict_results with field identifiers
        dict_results.update(_meta_params['field_identifiers'])
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the independent samples have identical average (expected) values.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results)
        return output_df


class TTestTwoSampleScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.ttest_rel"""

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            floats=['alpha'],
        )
        converted_params = add_default_params(converted_params, {'alpha': 0.05})
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        # pop 'alpha' from params into _meta_params and add the field identifiers
        _meta_params = {
            'field_identifiers': get_field_identifiers(a_array, "A_field", b_array, "B_field"),
            'alpha': self.params.pop('alpha'),
        }
        return a_array, b_array, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a scipy.stats 'Ttest_rel' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Update dict_results with field identifiers
        dict_results.update(_meta_params['field_identifiers'])
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the related or repeated samples have identical average (expected) values.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results)
        return output_df


class KSTest2SampleScoring(DoubleArrayScoringMixin, BaseScoring):
    """ Implements scipy.stats.ks_2samp """

    def handle_options(self, options):
        """ Ks_2samp requires each array to be made of exactly 1 field. """
        params = options.get('params', {})
        params = self.convert_param_types(params)
        a_fields, b_fields = get_and_check_fields_two_1d_arrays(options, self.scoring_name)
        return params, a_fields, b_fields

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            floats=['alpha'],
        )
        converted_params = add_default_params(converted_params, {'alpha': 0.05})
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, a_fields, b_fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields, b_fields)
        # Warn if the same field is being compared
        warn_on_same_fields(list(a_array.columns), list(b_array.columns))
        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        return a_array, b_array, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is scipy.stats 'Ks_2samp' object. """
        dict_results = {'statistic': result.statistic, 'p-value': result.pvalue}
        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the independent samples are drawn from the same continuous distribution.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class KPSSScoring(TSAStatsToolsMixin):
    """Implements statsmodels.tsa.stattools.kpss"""

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            strs=['regression'],
            ints=['lags'],
            floats=['alpha'],
        )
        converted_params = add_default_params(converted_params, {'regression': 'c', 'alpha': 0.05})
        converted_params = validate_param_from_str_list(converted_params, 'regression', ['c', 'ct'])
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields=fields, b_fields=None)
        # Set the default lags param
        maxlag = self.params.setdefault('lags',  int(np.ceil(12. * np.power(len(a_array) / 100., 1 / 4.))))
        if maxlag >= len(a_array):
            raise RuntimeError('Value error: parameter "lags" must be less than the number of samples.')

        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        return a_array, _meta_params

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a tuple. """
        dict_results = {
            'statistic': result[0], 'p-value': result[1], 'lags': result[2], 'critical values (1%)': result[3]['1%'],
            'critical values (2.5%)':  result[3]['2.5%'], 'critical values (5%)':  result[3]['5%'],
            'critical values (10%)':  result[3]['10%']
        }

        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'the field is level or trend stationary.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df


class AdfullerScoring(TSAStatsToolsMixin):
    """Implements statsmodels.tsa.stattools.adfuller"""

    @staticmethod
    def convert_param_types(params):
        converted_params = convert_params(
            params,
            strs=['autolag', 'regression'],
            ints=['maxlag'],
            floats=['alpha'],
        )
        converted_params = add_default_params(converted_params, {'autolag': 'AIC', 'regression': 'c', 'alpha': 0.05})
        converted_params = validate_param_from_str_list(converted_params, 'autolag', ['aic', 'bic', 't-stat', 'none'])
        converted_params = validate_param_from_str_list(converted_params, 'regression', ['c', 'ct', 'ctt', 'nc'])
        check_alpha_param(converted_params['alpha'])
        return converted_params

    def prepare_and_check_data(self, df, fields, params):
        a_array, b_array = prepare_statistical_scoring_data(df, a_fields=fields, b_fields=None)
        # Set the default maxlag param
        maxlag = self.params.setdefault('maxlag',  10)
        if maxlag >= len(a_array):
            raise RuntimeError('Value error: parameter "maxlag" must be less than the number of samples.')

        # pop 'alpha' from params into _meta_params
        _meta_params = {'alpha': self.params.pop('alpha')}
        return a_array, _meta_params

    def score(self, df, options):
        # Get preprocessed df and and meta-params for creating output
        preprocessed_df, _meta_params = self.prepare_and_check_data(df, self.variables, self.params)
        try:
            result = self.scoring_function(preprocessed_df.values.reshape(-1,), **self.params)
        except Exception:
            # Known bug when maxlag is close to sample size and autolag=t-stat
            # Perhaps other uncaught exceptions with adfuller method
            raise RuntimeError("Cannot compute 'adfuller' on sample. Try increasing the number of samples, reducing "
                               "the 'maxlag' parameter or modifying the value of autolag.")

        # Create output with meta-params
        df_output = self.create_output(self.scoring_name, result, _meta_params)
        return df_output

    def create_output(self, scoring_name, result, _meta_params=None):
        """ Result is a tuple. """
        dict_results = {
            'statistic': result[0], 'p-value': result[1], 'usedlag': result[2], 'nobs': result[3],
            'critical values (1%)': result[4]['1%'], 'critical values (5%)': result[4]['5%'],
            'critical values (10%)':  result[4]['10%']
        }
        if self.params['autolag'] is not None:
            dict_results['icbest'] = result[5]  # icbest is only available if autolag is not None

        # Annotate decision on whether to accept/reject null hypothesis
        null_hypothesis = 'there is a unit root.'
        dict_results = update_with_hypothesis_decision(dict_results, _meta_params['alpha'], null_hypothesis)
        output_df = pd.DataFrame(dict_results, index=[''])
        return output_df
