#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import inspect
import errno
import cexc
import models
from BaseProcessor import BaseProcessor

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class SummaryProcessor(BaseProcessor):
    """The summary processor calls the summary method of a saved model."""

    @staticmethod
    def load_model(model_name):
        """Try to load the model, error otherwise.

        Args:
            str: model name
        """
        try:
            algo_name, algo, model_options = models.load_model(model_name)
        except (OSError, IOError) as e:
            if e.errno == errno.ENOENT:
                raise RuntimeError('model "%s" does not exist.' % model_name)
            raise RuntimeError('Failed to load model "%s": %s.' % (
                model_name, str(e)))
        except Exception as e:
            cexc.log_traceback()
            raise RuntimeError('Failed to load model "%s": %s.' % (
                model_name, str(e)))
        return algo_name, algo, model_options

    @staticmethod
    def check_algo_has_summary(algo_name, algo):
        """Check if the algo supports summarization.

        Args:
            str: algo_name
            algo: loaded algo (from a model)
        Raises:
            RuntimeError
        """
        if not hasattr(algo, 'summary'):
            raise RuntimeError('"%s" models do not support summarization' % algo_name)

    @staticmethod
    def check_supports_params(algo):
        """Returns bool whether summary supports parameters such as json=t.

        Args:
            algo: loaded algo (from a model)
        Returns:
            boolean
        """
        return len(inspect.getargspec(algo.summary).args) > 1

    @staticmethod
    def get_summary(algo_name, algo, process_options):
        """Retrieve summary from the algorithm.

        Check first to see if params are allowed, error appropriately.
        Try to load a summary with options, then try a regular summary.

        Args:
            string: algo name
            algo: loaded algo (from a model)
            dict: process options
        Returns:
            dataframe
        """
        # summaries with options
        if SummaryProcessor.check_supports_params(algo):
            if 'args' in process_options:
                raise RuntimeError("Summarization does not take values other than parameters")
            try:
                df = algo.summary(options=process_options)
            except ValueError as e:
                raise RuntimeError(e)
        else:
            # normal summaries
            if any(opt in process_options for opt in ('args', 'params')):
                raise RuntimeError('"%s" models do not take options for summarization' % algo_name)
            df = algo.summary()
        return df

    def process(self):
        algo_name, algo, model_options = self.load_model(self.process_options['model_name'])
        self.check_algo_has_summary(algo_name, algo)
        self.df = self.get_summary(algo_name, algo, self.process_options)
