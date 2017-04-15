#!/usr/bin/env python

from collections import OrderedDict

from statsmodels.tsa.arima_model import ARIMA as _ARIMA
from statsmodels.tools.sm_exceptions import MissingDataError
import pandas as pd
import numpy as np
from base import TSMixin
from util.param_util import convert_params
from codec import codecs_manager
from codec.codecs import BaseCodec
import importlib
import cexc


# OUTPUT_NAME = 'prediction'


class ARIMA(TSMixin):
    def __init__(self, options):
        self.handle_options(options)

        params = convert_params(
            options.get('params', {}),
            strs=['order'],
            ints=['forecast_k', 'conf_interval'],
            aliases={'forecast_k': 'steps',
                     },
        )
        self.out_params = dict(model_params=dict(),
                               forecast_function_params=dict())

        if 'order' in params:
            # statsmodels wants a tuple for order of the model for the number of AR parameters,
            # differences, and MA parameters.
            # SPL won't accept a tuple as an option's value, so the next few lines will make it possible for the
            # user to configure order.
            try:
                self.out_params['model_params']['order'] = tuple(int(i) for i in params['order'].split('-'))
                assert len(self.out_params['model_params']['order']) == 3
            except:
                raise RuntimeError('Syntax Error: order requires three integer values, e.g. order=4-1-2')
        else:
            raise RuntimeError('Order of model is missing. It is required for fitting. e.g. order=<No. of AR>-'
                               '<Parameters-No. of Differences>-<No. of MA Parameters>')

        if 'steps' in params:
            self._test_forecast_k(params['steps'])
            self.out_params['forecast_function_params']['steps'] = params['steps']
        if 'conf_interval' in params:
            self.out_params['forecast_function_params']['alpha'] = \
                self._confidence_interval_to_alpha(params['conf_interval'])
        else:
            self.out_params['forecast_function_params'][
                'alpha'] = 0.05  # the default value that ARIMAResults.forecast uses.

        # Dealing with Missing data
        # if 'missing' in params and params['missing'] in ['raise', 'drop']:
        #     self.out_params['model_params']['missing'] = params['missing']
        # else:
        self.out_params['model_params']['missing'] = 'raise'

    @staticmethod
    def _test_forecast_k(x):
        if x < 1:
            raise RuntimeError('forecast_k should be an integer equal or greater than 1.')

    @staticmethod
    def _confidence_interval_to_alpha(x):
        """"
        Transforming confidence interval to alpha for ARIMAResults.forecast
        """
        if x >= 100 or x <= 0:
            raise RuntimeError('conf_interval cannot be less than 0 or more than 100.')
        return 1 - x / 100.0

    @staticmethod
    def _alpha_to_confidence_interval(x):
        """"
        Transforming alpha to confidence interval
        """
        return int(round((1 - x) * 100))

    @staticmethod
    def _find_freq(X, threshold=0.9):
        """
        Calculates the dominant value of differences between two consequent timestamps.
        Checks if its frequency is above the threshold (default=.90)
        """

        y = np.diff(X)
        median = np.median(y)
        ratio = np.mean(y == median)
        if ratio < threshold:
            raise RuntimeError('Sampling is irregular. Try excluding _time from \'fit\'.')

        return median

    @staticmethod
    def _generate_timestamps_for_forecast(forecast_k, freq, last_timestamp):
        return np.arange(1, forecast_k + 1) * freq + last_timestamp

    def _fit(self, X):
        for variable in self.variables:
            self.assert_field_present(X, variable)
        self.drop_unused_fields(X, self.variables)
        self.assert_any_fields(X)
        self.assert_any_rows(X)

        if X[self.time_series].dtype == object:
            raise ValueError('%s contains non-numeric data. ARIMA only accepts numeric data.' % self.time_series)
        X[self.time_series] = X[self.time_series].astype(float)

        try:
            self.estimator = _ARIMA(X[self.time_series].values,
                                    order=self.out_params['model_params']['order'],
                                    missing=self.out_params['model_params']['missing']).fit(disp=False)
        except ValueError as e:
            if 'stationary' in e.message:
                raise ValueError("The computed initial AR coefficients are not "
                                 "stationary. You should induce stationarity by choosing a different model order.")
            elif 'invertible' in e.message:
                raise ValueError("The computed initial MA coefficients are not invertible. "
                                 "You should induce invertibility by choosing a different model order.")
            else:
                cexc.log_traceback()
                raise ValueError(e)
        except MissingDataError:
            raise RuntimeError('Empty or null values are not supported in %s. '
                               'If using timechart, try using a larger span.'
                               % self.time_series)
        except Exception as e:
            cexc.log_traceback()
            raise RuntimeError(e)

        # Saving the _time but not as a part of the ARIMA structure but as new attribute for ARIMA.
        if '_time' in self.variables:
            self.estimator.datetime_information = dict(ver=0,
                                                       _time=X['_time'].values,
                                                       freq=self._find_freq(X['_time'].values),
                                                       # in seconds (unix epoch)
                                                       first_timestamp=X['_time'].values[0],
                                                       last_timestamp=X['_time'].values[-1],
                                                       length=len(X))
        else:
            self.estimator.datetime_information = dict(ver=0,
                                                       _time=None,
                                                       freq=None,
                                                       first_time=None,
                                                       last_time=None,
                                                       length=len(X))

    def _forecast(self, options, output_name=None):
        forecast_output = self.estimator.forecast(**options)

        # output_name = 'Predicted(%s)' % self.time_series
        if output_name is None:
            output_name = 'predicted(%s)' % self.time_series
        # output_name = OUTPUT_NAME
        lower_name = 'lower%d(%s)' % (self._alpha_to_confidence_interval(options['alpha']), output_name)
        upper_name = 'upper%d(%s)' % (self._alpha_to_confidence_interval(options['alpha']), output_name)
        # std_err = 'standard_error'

        output = pd.DataFrame(columns=[self.time_series, output_name, lower_name, upper_name],
                              index=range(self.estimator.datetime_information['length'],
                                          self.estimator.datetime_information['length'] + options['steps']))
        output[output_name] = forecast_output[0]
        # output[std_err] = forecast_output[1]
        output[lower_name] = forecast_output[2][:, 0]
        output[upper_name] = forecast_output[2][:, 1]
        if self.estimator.datetime_information['ver'] == 0 and self.estimator.datetime_information['freq'] is not None:
            output['_time'] = self._generate_timestamps_for_forecast(options['steps'],
                                                                     self.estimator.datetime_information['freq'],
                                                                     self.estimator.datetime_information[
                                                                         'last_timestamp'])
        return output

    def fit(self, X):
        self._fit(X)

    def fit_predict(self, X, options=None, output_name=None):
        self._fit(X)

        # output_name = OUTPUT_NAME
        if output_name is None:
            output_name = 'predicted(%s)' % self.time_series

        output = X.copy()
        output[output_name] = np.nan
        output[output_name][self.out_params['model_params']['order'][1]:] = self.estimator.predict()

        if 'steps' in self.out_params['forecast_function_params']:
            forecast_output = self._forecast(self.out_params['forecast_function_params'], output_name)
            extra_columns = set(forecast_output.columns).difference(output)
            for col in extra_columns:
                output[col] = np.nan
            output = output.append(forecast_output)

        return {'append': False,
                'output': output.sort(ascending=True)}

    # TODO: In its current form, predict does not use X. Thus, this
    # algo can't be used to forecast on out-of-sample time-series
    # data. We disable predict() (and by extension, saved models and
    # the apply command) for now.
    def __disabled_predict(self, X, options=None, output_name=None):
        params = convert_params(
            options.get('params', {}),
            ints=['forecast_k', 'conf_interval'],
            bools=['show_past'],
            aliases={'forecast_k': 'steps',
                     'conf_interval': 'alpha'
                     },
            ignore_extra=True
        )
        if 'alpha' in params:
            params['alpha'] = self._confidence_interval_to_alpha(params['alpha'])
        else:
            params['alpha'] = self.out_params['forecast_function_params']['alpha']
        if 'steps' not in params:
            params['steps'] = 1
        else:
            self._test_forecast_k(params['steps'])

        show_past = params.get('show_past')
        if show_past is not None:
            params.pop('show_past')
        else:
            show_past = False

        if output_name is None:
            output_name = 'predicted(%s)' % self.time_series

        output = self._forecast(params, output_name)
        if show_past:
            history_output = pd.DataFrame(self.estimator.data.endog, columns=[self.time_series])
            if self.estimator.datetime_information['_time'] is not None:
                history_output['_time'] = self.estimator.datetime_information['_time']
            # output = self.estimator.copy()
            history_output[output_name] = np.nan
            history_output[output_name][self.out_params['model_params']['order'][1]:] = self.estimator.predict()

            extra_columns = set(output.columns).difference(history_output)
            for col in extra_columns:
                history_output[col] = np.nan
            output = history_output.append(output)

        return {'append': False,
                'output': output.sort(ascending=False)}

    def summary(self):
        summary = OrderedDict()  # dict()
        attrs = dir(self.estimator)

        summary['k_ar'] = self.estimator.k_ar if 'k_ar' in attrs else 0
        summary['k_ma'] = self.estimator.k_ma if 'k_ma' in attrs else 0
        summary['k_diff'] = self.estimator.k_diff if 'k_diff' in attrs else 0

        summary['model'] = 'ARIMA(%d, %d, %d)' % (summary['k_ar'], summary['k_diff'], summary['k_ma'])

        if 'params' in attrs:
            summary['constant'] = self.estimator.params[0]

        if 'arparams' in attrs:
            for i, param in enumerate(self.estimator.arparams, 1):
                key = 'ar_lag_%d' % i
                summary[key] = param

        if 'maparams' in attrs:
            for i, param in enumerate(self.estimator.maparams, 1):
                key = 'ma_lag_%d' % i
                summary[key] = param

        if 'arroots' in attrs:
            for i, val, in enumerate(np.real(self.estimator.arroots), 1):
                key = 'AR_Real_%d' % i
                summary[key] = val
            for i, val in enumerate(np.imag(self.estimator.arroots), 1):
                key = 'AR_Imag_%d' % i
                summary[key] = val

        if 'maroots' in attrs:
            for i, val, in enumerate(np.real(self.estimator.maroots), 1):
                key = 'MA_Real_%d' % i
                summary[key] = val
            for i, val in enumerate(np.imag(self.estimator.maroots), 1):
                key = 'MA_Imag_%d' % i
                summary[key] = val

        if 'aic' in attrs:
            summary['AIC'] = self.estimator.aic
        if 'bic' in attrs:
            summary['BIC'] = self.estimator.bic
        if 'hqic' in attrs:
            summary['HQIC'] = self.estimator.hqic
        if 'llf' in attrs:
            summary['LLF'] = self.estimator.llf
        if 'sigma2' in attrs:
            summary['Residual_Variance'] = self.estimator.sigma2

        return pd.DataFrame(summary.items(), columns=['features', 'coefficients'])

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('statsmodels.tsa.arima_model', 'ARIMAResultsWrapper', SimpleObjectCodec)
        codecs_manager.add_codec('statsmodels.tsa.arima_model', 'ARMAResultsWrapper', SimpleObjectCodec)
        codecs_manager.add_codec('statsmodels.tsa.arima_model', 'ARMAResults', ARMAResultsCodec)
        codecs_manager.add_codec('statsmodels.tsa.arima_model', 'ARIMAResults', ARIMAResultsCodec)
        codecs_manager.add_codec('statsmodels.tsa.arima_model', 'ARMA', SimpleObjectCodec)
        codecs_manager.add_codec('statsmodels.tsa.arima_model', 'ARIMA', ARIMACodec)
        codecs_manager.add_codec('statsmodels.base.model', 'LikelihoodModelResults', SimpleObjectCodec)
        codecs_manager.add_codec('statsmodels.base.data', 'ModelData', SimpleObjectCodec)
        codecs_manager.add_codec('algos.ARIMA', 'ARIMA', SimpleObjectCodec)


class ARMAResultsCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__

        components = obj.__dict__
        components.pop('model', None)
        components.pop('data', None)

        return {
            '__mlspl_type': [module, name],
            'dict': components,
        }

    @classmethod
    def decode(cls, obj):
        module_name, name = obj['__mlspl_type']

        module = importlib.import_module(module_name)
        class_ref = getattr(module, name)

        new_obj = class_ref.__new__(class_ref)
        new_obj.__dict__ = obj['dict']
        new_obj.__dict__['model'] = new_obj.mlefit.model
        new_obj.__dict__['data'] = new_obj.mlefit.model.data

        return new_obj


class ARIMAResultsCodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__

        components = obj.__dict__
        components.pop('data', None)

        return {
            '__mlspl_type': [module, name],
            'dict': components,
        }

    @classmethod
    def decode(cls, obj):
        module_name, name = obj['__mlspl_type']

        module = importlib.import_module(module_name)
        class_ref = getattr(module, name)

        new_obj = class_ref.__new__(class_ref)
        new_obj.__dict__ = obj['dict']
        new_obj.__dict__['data'] = new_obj.model.data

        return new_obj


class ARIMACodec(BaseCodec):
    @classmethod
    def encode(cls, obj):
        name, module = type(obj).__name__, type(obj).__module__

        components = obj.__dict__

        return {
            '__mlspl_type': [module, name],
            'dict': components,
        }

    @classmethod
    def decode(cls, obj):
        obj_dict = obj['dict']

        from statsmodels.tsa.arima_model import ARIMA
        new_obj = ARIMA(obj_dict['data'].endog,
                        dates=obj_dict['data'].dates,
                        order=(obj_dict['k_ar'], obj_dict['k_diff'], obj_dict['k_ma']))
        new_obj.exog = obj_dict['exog']
        # new_obj.data._cache['xnames'] = obj_dict['data']._cache['xnames']
        new_obj.k_trend = obj_dict['k_trend']
        new_obj.method = obj_dict['method']
        new_obj.nobs = obj_dict['nobs']
        new_obj.sigma2 = obj_dict['sigma2']
        new_obj.transparams = obj_dict['transparams']

        return new_obj
