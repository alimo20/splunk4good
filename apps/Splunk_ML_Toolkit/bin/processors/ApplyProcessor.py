#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import errno
import pandas as pd
import gc

import cexc
import conf
import models
from BaseProcessor import BaseProcessor

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class ApplyProcessor(BaseProcessor):
    """The apply processor receives and returns pandas DataFrames."""

    def __init__(self, process_options):
        """Initialize options for the processor.

        Args:
            dict: process options
        """
        self.algo_name, self.algo, self.process_options = self.setup_model(process_options)
        self.resource_limits = self.load_resource_limits(self.algo_name)

    def get_relevant_fields(self):
        """Return the needed explanatory variables/variables.
        If explanatory variables are present, that's all we need,
        otherwise we need variables

        Returns:
            list: relevant fields
        """
        if 'explanatory_variables' in self.process_options:
            return self.process_options['explanatory_variables']
        else:
            return self.process_options['variables']

    @staticmethod
    def setup_model(process_options):
        """Load temp model, try to load real model, update options.

        Args:
            dict: process_options
        Returns:
            tuple: the returned tuple contains four members:
                str: algo_name
                algo: algorithm object
                dict: updated process options
        """
        # Try loading tmp model from dispatch directory.
        try:
            algo_name, _, model_options = models.load_model(
                process_options['model_name'],
                model_dir=process_options['tmp_dir'],
                skip_model_obj=True,
                tmp=True
            )
            algo = None
            logger.debug('Using tmp model to set required_fields.')
        except:
            # Try to load real model.
            try:
                algo_name, algo, model_options = models.load_model(process_options['model_name'])
            except (OSError, IOError) as e:
                if e.errno == errno.ENOENT:
                    raise RuntimeError('model "%s" does not exist.' % process_options['model_name'])
                raise RuntimeError('Failed to load model "%s": %s.' % (
                    process_options['model_name'], str(e)))
            except Exception as e:
                cexc.log_traceback()
                raise RuntimeError('Failed to load model "%s": %s.' % (
                    process_options['model_name'], str(e)))

        model_options.update(process_options)  # process options override loaded model options
        process_options = model_options
        return algo_name, algo, process_options

    @staticmethod
    def load_resource_limits(algo_name):
        """Load algorithm-specific limits.

        Args:
            str: algo name
        Returns:
            dict: dictionary of resource limits
        """
        resource_limits = {}
        resource_limits['max_memory_usage_mb'] = int(
            conf.get_mlspl_prop('max_memory_usage_mb', algo_name, -1))
        return resource_limits

    @staticmethod
    def apply(df, algo, process_options):
        """Perform the literal predict from the estimator.

        Args:
            dataframe: input data
            algo: initialized algo
            dict: process options
        Returns:
            dataframe: output data
        """
        try:
            output_name = process_options.get('output_name', None)
            prediction_df = algo.predict(df.copy(), options=process_options,
                                         output_name=output_name)
            gc.collect()

        except Exception as e:
            cexc.log_traceback()
            cexc.messages.warn('Error while applying model "%s": %s' % (process_options['model_name'], str(e)))
            raise RuntimeError(e)

        # TODO: ARIMA should not get to be special and affect any of the processor logic here:
        if type(prediction_df) is dict and prediction_df['append'] is False:
            df = prediction_df['output'].copy()
        else:
            df.drop(prediction_df.columns, axis=1, errors='ignore', inplace=True)
            df = pd.concat([df, prediction_df], axis=1, join_axes=[df.index])

        return df

    def process(self):
        """If algo isn't loaded, load the model. Create the output dataframe."""
        if self.algo is None:
            self.algo_name, self.algo, _ = models.load_model(self.process_options['model_name'])
        if len(self.df) > 0:
            self.df = self.apply(self.df, self.algo, self.process_options)
