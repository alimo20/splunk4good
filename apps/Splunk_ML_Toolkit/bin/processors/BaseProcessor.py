#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import cexc

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class BaseProcessor(object):
    """Skeleton for all processors, also implements some utility methods."""

    def __init__(self, process_options=None):
        """Pass process_options.

        Args:
            dict: process options
        """
        self.process_options = process_options

    def receive_input(self, df):
        """Get dataframe.

        Args:
            dataframe: input dataframe
        """
        self.df = df

    def process(self):
        """Necessary process method."""
        raise NotImplementedError

    def get_output(self):
        """Simply return the output dataframe.

        Returns:
            dataframe: output dataframe
        """
        return self.df
