#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import pandas as pd

import cexc
import conf
import models
from sampler import ReservoirSampler
from util.param_util import is_truthy, convert_params
from util.base_util import is_valid_identifier
from BaseProcessor import BaseProcessor

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class FitBatchProcessor(BaseProcessor):
    """The fit batch processor receives and returns pandas DataFrames."""

    def __init__(self, process_options):
        """Initialize options for processor.

        Args:
            dict: process options
        """
        # Split apart process & algo options
        # TODO: remove self.algo_options, and to delegate all uses of algo_options to the algo itself
        self.process_options, self.algo_options = self.split_options(process_options)

        # Convenience / readability
        do_apply = self.process_options['do_apply']
        self.tmp_dir = self.process_options['tmp_dir']

        self.algo = self.initialize_algo(self.algo_options)
        self.check_algo_options(do_apply, self.algo_options, self.algo)
        self.save_temp_model(self.algo_options, self.tmp_dir)
        self.resource_limits = self.load_resource_limits(self.algo_options['algo_name'])

        self._sampler_time = 0.0
        self.sampler_limits = self.load_sampler_limits(self.process_options, self.algo_options['algo_name'])
        self.sampler = self.get_sampler(self.sampler_limits)

    @staticmethod
    def split_options(options):
        """Pop do_apply and tmp_dir from the options. Also, parse sample count
        and sample seed from original params and add them to process options.

        Args:
            dict: process options

        Returns:
            tuple: the returned tuple contains two dictionaries
                - the process options we use here
                - the algo options passed to the algorithm
         """
        sample_params = {}
        if 'params' in options:
            try:
                sample_params = convert_params(options['params'],
                                               ignore_extra=True,
                                               ints=['sample_count',
                                                     'sample_seed']
                                               )
            except ValueError as e:
                raise RuntimeError(str(e))

        if 'sample_count' in sample_params:
            del options['params']['sample_count']

        if 'sample_seed' in sample_params:
            del options['params']['sample_seed']

        # copy everything from leftover options to algo options
        algo_options = options.copy()

        # brand new process options
        process_options = {}

        # sample options are added to the process options
        process_options['sample_seed'] = sample_params.get('sample_seed', None)
        process_options['sample_count'] = sample_params.get('sample_count', None)

        # needed by processor, not algorithm
        process_options['tmp_dir'] = algo_options.pop('tmp_dir')
        process_options['do_apply'] = algo_options.pop('do_apply')

        return process_options, algo_options

    @staticmethod
    def initialize_algo(algo_options):
        """Import and initialize the algorithm.

        Args:
            dict: algo_options
        Returns:
            object: initialized algo
        """
        try:
            assert is_valid_identifier(algo_options['algo_name'])
            algos = __import__("algos", fromlist=["*"])
            algo_module = getattr(algos, algo_options['algo_name'])
            algo_class = getattr(algo_module, algo_options['algo_name'])
        except (AttributeError, AssertionError) as e:
            raise RuntimeError('Failed to load algorithm "%s"' % algo_options['algo_name'])
        try:
            algo = algo_class(algo_options)
        except Exception as e:
            cexc.log_traceback()
            raise RuntimeError('Error while initializing algorithm "%s": %s' % (
                algo_options['algo_name'], str(e)))
        return algo

    @staticmethod
    def check_algo_options(do_apply, algo_options, algo):
        """Raise errors if options are incompatible

        Args:
            bool: do_apply flag
            dict: algo_options
            algo: initialized algo
        Raises:
            RuntimeError
        """

        # Ensure model present if no apply
        if not do_apply and 'model_name' not in algo_options:
            raise RuntimeError('You must save a model if you are not applying it.')
        # TODO: avoid hasattr
        # Pre-validate whether or not this algo supports saved models.
        if 'model_name' in algo_options and not hasattr(algo, 'predict'):
            raise RuntimeError('Algorithm "%s" does not support saved models' % algo_options['algo_name'])

    @staticmethod
    def fit(df, algo, algo_options):
        """Perform the literal fitting process, and predict if fit_predict.

        Some of the algorithms only support a fit_predict method. This means
        that they cannot predict independently of fitting the model. Thus, for
        algorithms that do not have fit_predict, we set prediction_df to an empty
        dataframe that we will update with a predict method later.

        Args:
            dataframe: dataframe to work with
            algo: loaded algo
            dict: algo_options
        Returns:
            algo: updated algo object
            dataframe: output dataframe
        """
        try:
            # TODO: avoid hasattr
            if hasattr(algo, 'fit_predict'):
                output_name = algo_options.get('output_name', None)
                prediction_df = algo.fit_predict(df.copy(), options=algo_options, output_name=output_name)
            else:
                algo.fit(df.copy())
                # empty output for now
                prediction_df = pd.DataFrame()
        except Exception as e:
            cexc.log_traceback()
            raise RuntimeError('Error while fitting "%s" model: %s' % (algo_options['algo_name'], str(e)))

        return algo, prediction_df

    @staticmethod
    def load_sampler_limits(process_options, algo_name):
        """Read sampling limits from conf file and decide sample count.

        Args:
            dict: process options
            str: algo name
        Returns:
            dict: sampler limits
        """
        sampler_limits = {}

        # setting up the logic to choose the sample count
        sampler_limits['use_sampling'] = is_truthy(str(conf.get_mlspl_prop('use_sampling', algo_name, 'yes')))
        max_inputs = int(conf.get_mlspl_prop('max_inputs', algo_name, -1))
        if process_options['sample_count']:
            sampler_limits['sample_count'] = min(process_options['sample_count'], max_inputs)
        else:
            sampler_limits['sample_count'] = max_inputs

        # simply set sample seed
        sampler_limits['sample_seed'] = process_options['sample_seed']
        return sampler_limits

    @staticmethod
    def get_sampler(sampler_limits):
        """Initialize the sampler and use resource limits from processor.

        Args:
            dict: sampler limits
        Returns:
            sampler: sampler object
        """
        return ReservoirSampler(sampler_limits['sample_count'], random_state=sampler_limits['sample_seed'])

    @staticmethod
    def check_sampler(sampler_limits, algo_name):
        """Inform user if sampling is on or raise error if sampling is off and events exceed limit.

        Args:
            dict: sampler limits
            str: algo name
        """
        if is_truthy(sampler_limits['use_sampling']):
            messages.warn(
                'Input event count exceeds max_inputs for %s (%d), model will be fit on a sample of events.' % (
                    algo_name, sampler_limits['sample_count']))
        else:
            raise RuntimeError('Input event count exceeds max_inputs for %s (%d) and sampling is disabled.' % (
                algo_name, sampler_limits['sample_count']))

    @staticmethod
    def load_resource_limits(algo_name):
        """Load algorithm specific resource limits.

        Args:
            str: algo_name

        Returns:
            dictionary of resource limits including max_fit_time, max_memory_usage_mb, and max_model_size_mb
        """
        resource_limits = {}
        resource_limits['max_memory_usage_mb'] = int(conf.get_mlspl_prop('max_memory_usage_mb', algo_name, -1))
        resource_limits['max_fit_time'] = int(conf.get_mlspl_prop('max_fit_time', algo_name, -1))
        resource_limits['max_model_size_mb'] = int(conf.get_mlspl_prop('max_model_size_mb', algo_name, -1))
        return resource_limits

    @staticmethod
    def save_temp_model(algo_options, tmp_dir):
        """Save temp model for follow-up apply.

        Args:
            dict: algo options
            str: temp directory to save model to
        """
        if 'model_name' in algo_options:
            try:
                models.save_model(algo_options['model_name'], None,
                                  algo_options['algo_name'], algo_options,
                                  model_dir=tmp_dir, tmp=True)
            except Exception as e:
                cexc.log_traceback()
                raise RuntimeError(
                    'Error while saving temporary model "%s": %s' % (algo_options['model_name'], e))

    def get_relevant_fields(self):
        """Ask algo for relevant variables and return as relevant fields.

        Returns:
            list: relevant fields
        """
        relevant_fields = self.algo.get_relevant_variables()
        return relevant_fields

    def save_model(self):
        """Attempt to save the model, delete the temporary model."""
        if 'model_name' in self.algo_options:
            try:
                models.save_model(self.algo_options['model_name'], self.algo,
                                  self.algo_options['algo_name'], self.algo_options,
                                  max_size=self.resource_limits['max_model_size_mb'])
            except Exception as e:
                cexc.log_traceback()
                raise RuntimeError('Error while saving model "%s": %s' % (self.algo_options['model_name'], e))
            try:
                models.delete_model(self.algo_options['model_name'], model_dir=self.tmp_dir,
                                    tmp=True)
            except Exception as e:
                cexc.log_traceback()
                logger.warn('Exception while deleting tmp model "%s": %s', self.algo_options['model_name'], e)

    def receive_input(self, df):
        """Receive dataframe and append to sampler if necessary.

        Args:
            dataframe: dataframe we receive from controller
        """
        if self.sampler_limits['sample_count'] - len(df) < self.sampler.count <= self.sampler_limits['sample_count']:
            self.check_sampler(sampler_limits=self.sampler_limits, algo_name=self.algo_options['algo_name'])

        with cexc.Timer() as sampler_t:
            self.sampler.append(df)
        self._sampler_time += sampler_t.interval

        logger.debug('sampler_time=%f', sampler_t.interval)

    def process(self):
        """Get dataframe, update algo, and possibly make predictions."""
        self.df = self.sampler.get_df()
        self.algo, self.prediction_df = self.fit(self.df, self.algo, self.algo_options)

    def get_output(self):
        """Override get_output from BaseProcessor.

        Check if algo has fit_predict method and already made
        prediction, otherwise make prediction.

        Check for and handle special ARIMA behavior.

        Returns:
            dataframe: output
        """
        output_name = self.algo_options.get('output_name', None)

        if len(self.prediction_df) == 0:
            self.prediction_df = self.algo.predict(self.df.copy(),
                                                   options=self.algo_options,
                                                   output_name=output_name
                                                   )

        # TODO: ARIMA should not get to be special and affect any of the processor logic here:
        if type(self.prediction_df) is dict and self.prediction_df['append'] is False:
            self.df = self.prediction_df['output']
        else:
            self.df.drop(self.prediction_df.columns,
                         axis=1, errors='ignore',
                         inplace=True
                         )

            self.df = pd.concat([self.df, self.prediction_df],
                                axis=1,
                                join_axes=[self.df.index]
                                )
        return self.df
