#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import csv
import importlib
import pandas as pd
from cStringIO import StringIO

import cexc
import models

from util.base_util import match_field_globs

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class ChunkedController:
    """The controller connects CEXC commands to ML-SPL processors.

    - abstracting our interaction with the CEXC protocol and calling command
    - converting CSVs to pandas DataFrames
    - initializing and calling the processor's process method
    - notifying rest end point after a model is saved
    """

    def __init__(self, getinfo, controller_options):
        """Set a variety of attributes on self, call initializer methods.

        Args:
            dict: getinfo metadata
            dict: controller options passed from caller
        """
        # Attributes
        self._csv_parse_time = 0.0
        self._csv_render_time = 0.0

        # Parse options
        self.getinfo, self.controller_options, process_options = self.split_options(controller_options, getinfo)

        # initializer methods
        self.processor = self.initialize_processor(self.controller_options['processor'], process_options)
        self.resource_limits = self.get_resource_limits(self.processor)

    @staticmethod
    def split_options(options, getinfo):
        """Split apart controller and processor options.

        Args:
            dict: controller options
            dict: getinfo metadata
        Returns:
            tuple: the returned tuple contains 3 dictionaries
                - the get info dictionary we passed in
                - the controller options we will use here
                - the process options we pass to the processor
        """
        # Default value
        controller_options = {}
        controller_options['use_processor_output'] = options.get('apply', True)
        controller_options['processor'] = options.pop('processor')
        if 'model_name' in options:
            controller_options['model_name'] = options.get('model_name')

        # Construct process option
        process_options = options
        process_options['do_apply'] = options.get('apply', True)
        process_options['tmp_dir'] = getinfo['searchinfo']['dispatch_dir']

        return getinfo, controller_options, process_options

    @staticmethod
    def initialize_processor(processor_name, process_options):
        """Import and initialize a processor.

        Processors are stored in ./bin/processors/
        The processors all inherit from the BaseProcessor class.

        Args:
            str: processor name
            dict: process options
        Returns:
            object: initialized processor
        """
        try:
            processor_module = importlib.import_module('processors.{}'.format(processor_name))
            processor_class = getattr(processor_module, processor_name)
        except AttributeError as e:
            logger.debug('Failed to import ML-SPL processor "%s"' % processor_name)
            raise RuntimeError('Failed to import ML-SPL processor.')
        try:
            processor = processor_class(process_options)
        except Exception as e:
            cexc.log_traceback()
            logger.debug('Error while initializing processor "%s": %s' % (
                processor_name, str(e)))
            raise RuntimeError(str(e))
        return processor

    @staticmethod
    def parse_relevant_fields(sio, relevant_fields):
        """Get relevant fields from processor & parse them from StringIO to dataframe.

        Args:
            StringIO: buffer
            list of str: list of field names
        Returns:
            dataframe
        """
        dict_reader = csv.DictReader(sio)
        input_fields = dict_reader.fieldnames
        fields_to_parse = match_field_globs(input_fields, relevant_fields)
        sio.seek(0)
        df = pd.read_csv(sio, usecols=[str(f) for f in fields_to_parse])
        return df

    def get_required_fields(self):
        """Fetch required fields from processor's relevant fields.

        Required fields are sent back during the getinfo exchange. This is how
        Splunk knows which fields will be needed by our command and is akin
        to setting required fields with the SPL fields command.

        Returns:
            list: list of required fields
        """
        required_fields = self.processor.get_relevant_fields()
        return required_fields

    @staticmethod
    def get_resource_limits(processor):
        """Fetch resource limits from the processor.

        Args:
            object: initialized processor
        Returns:
            dict: resource limits with keys such as max_memory_usage
        """
        try:
            resource_limits = processor.resource_limits
        except AttributeError:
            logger.debug('No resource limits were loaded for ML-SPL processor "%s"' % processor)
            resource_limits = None
        return resource_limits

    def load_data(self, body):
        """Load fields from search results into a DataFrame.

        In addition to literally converting the body into a data frame, note that
        the processor also receives the dataframe here.

        Args:
            str: csv body chunk from CEXC process
        """
        self.body = body
        if len(body) != 0:
            sio = StringIO(body)

            with cexc.Timer() as csv_t:
                if self.controller_options['use_processor_output']:
                    df = pd.read_csv(sio)
                else:
                    relevant_fields = self.get_required_fields()
                    df = self.parse_relevant_fields(sio, relevant_fields)
            self._csv_parse_time += csv_t.interval
            logger.debug('chunk body: %d bytes, %d rows, %d columns, csv_read_time=%f',
                         len(body), len(df), len(df.columns), csv_t.interval)

            self.processor.receive_input(df)

    def execute(self):
        """Call the processor's process method."""
        self.processor.process()

    def output_results(self):
        """Get processor's output and convert to CSV string if do_apply.

        If do_apply is set to false, then the body will not be converted.

        Returns:
            string: body, a CSV data payload for CEXC
        """
        with cexc.Timer() as csv_t:
            if self.controller_options['use_processor_output']:
                body = self.processor.get_output().to_csv(index=False)
            else:
                body = self.body
        self._csv_render_time += csv_t.interval
        return body

    def finalize(self):
        """Save model and notify REST endpoint about new lookup."""
        if "model_name" in self.controller_options:
            self.processor.save_model()
            models.notify_model_update(self.getinfo, self.controller_options['model_name'])
