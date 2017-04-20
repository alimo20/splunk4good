#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import errno
import pandas as pd

import cexc
import conf
import models

from FitBatchProcessor import FitBatchProcessor

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class FitPartialProcessor(FitBatchProcessor):
    """The fit partial processor receives and returns pandas DataFrames.

    This processor inherits from FitBatchProcessor and uses a handful of its
    methods. The partial processor does not need sampling and has a few
    additional things it needs to keep track of, including:
        - attempting to load a model
        - checking for discrepancies between search & saved model
        - handling new categorical values as specified by the unseen_value param
    """

    def __init__(self, process_options):
        """Initialize options for processor.

        Args:
            dict: process options
        """
        # Split apart process & algo options
        self.process_options, self.algo_options = self.split_options(process_options)

        # TODO: delegate this to the relevant algorithms
        self.process_options['handle_new_cat'] = self.get_unseen_value_behavior(process_options)

        # Convenience / readability
        self.tmp_dir = self.process_options['tmp_dir']

        # Try load algo from a saved model
        self.algo = self.initialize_algo_from_model(self.algo_options)
        if self.algo is None:
            # Initialize algo from scratch
            self.algo = self.initialize_algo(self.algo_options)
            # check if this algo supports partial_fit
            self.check_algo_options(self.algo, self.algo_options)
        else:
            # Check if the loaded model supports partial_fit
            self.check_algo_options(self.algo, self.algo_options)
            # Warn
            self.warn_about_new_parameters()

        self.save_temp_model(self.algo_options, self.tmp_dir)
        self.resource_limits = self.load_resource_limits(self.algo)

    @staticmethod
    def get_unseen_value_behavior(process_options):
        """Parse unseen_value if present & determine unseen_value behavior.

        Args:
            dict: process options

        Returns:
            str: string describing how to handle new categorical values during
                a partial fit. The values can be seen in mlspl.conf
        """

        handle_new_cat = conf.get_mlspl_prop('handle_new_cat', stanza='default', default='default')

        if 'params' in process_options:
            if process_options['params'].get('unseen_value', []):
                handle_new_cat = process_options['params']['unseen_value']
                del process_options['params']['unseen_value']

        return handle_new_cat

    @staticmethod
    def initialize_algo_from_model(algo_options):
        """Init algo from model if possible, and catch discrepancies.

        Args:
            dict: algo options
        Returns:
            algo: initialized algo
        """
        if 'model_name' in algo_options:
            try:
                model_algo_name, algo, model_options = models.load_model(
                    algo_options['model_name'])
            except (OSError, IOError) as e:
                if e.errno == errno.ENOENT:
                    # No existing model with matching name found
                    algo = None
                else:
                    raise RuntimeError('Failed to load model "%s". Error: %s.' % (
                        algo_options['model_name'], str(e)))
            except Exception as e:
                cexc.log_traceback()
                raise RuntimeError('Failed to load model "%s". Exception: %s.' % (
                    algo_options['model_name'], str(e)))

            if algo is not None:
                FitPartialProcessor.catch_model_discrepancies(algo_options,
                                                              model_options,
                                                              model_algo_name)
            return algo

    @staticmethod
    def warn_about_new_parameters():
        cexc.messages.warn(
            'Partial fit on existing model ignores newly supplied parameters. '
            'Parameters supplied at model creation are used instead')

    @staticmethod
    def catch_model_discrepancies(algo_options, model_options, model_algo_name):
        """Check to see if algo name or input columns are different from the model.

        Args:
            dict: algo options
            dict: model options
            str: name of algo from loaded model
        """
        # Check for discrepancy between algorithm name of the model loaded and algorithm name specified in input
        try:
            assert (algo_options['algo_name'] == model_algo_name)
        except AssertionError:
            raise RuntimeError("Model was trained using algorithm %s but found %s in input" % (
                model_algo_name, algo_options['algo_name']))

        # Check for discrepancy between model columns and input columns
        if (model_options['variables'] != algo_options.get('variables', []) or
                ('explanatory_variables' in algo_options and
                         model_options['explanatory_variables'] != algo_options.get(
                         'explanatory_variables', []))):
            raise RuntimeError("Model was trained on data with different columns than given input")

    @staticmethod
    def check_algo_options(algo, algo_options):
        """Validate processor options.

        Args:
            algo: initialized algo
            dict: algo options
        """
        if not hasattr(algo, 'partial_fit'):
            raise RuntimeError('Algorithm "%s" does not support partial fit' % algo_options['algo_name'])

        if 'model_name' not in algo_options:
            raise RuntimeError('You must save a model if you fit the model with partial_fit enabled')

    @staticmethod
    def fit(algo, df, handle_new_cat):
        """Perform the partial fit.

        Args:
            algo: algo object
            dataframe: dataframe to fit on
            str: handle new cat value for partial fit
        Returns:
            algo: updated algorithm
        """
        algo.partial_fit(df.copy(), handle_new_cat)
        return algo

    def receive_input(self, df):
        """Override batch processor, simply pass df to self.

        Args:
            dataframe: dataframe to receive
        """
        self.df = df

    def process(self):
        """Run fit and update algo."""
        self.algo = self.fit(self.algo,
                             self.df,
                             self.process_options['handle_new_cat'])

    def get_output(self):
        """Predict if necessary & return appropriate dataframe.

        Returns:
            dataframe: output dataframe
        """
        output_name = self.algo_options.get('output_name', None)
        prediction_df = self.algo.predict(self.df.copy(),
                                          options=self.algo_options,
                                          output_name=output_name
                                          )
        self.df.drop(prediction_df.columns,
                     axis=1, errors='ignore',
                     inplace=True
                     )
        self.df = pd.concat([self.df, prediction_df],
                            axis=1, join_axes=[self.df.index]
                            )
        return self.df
