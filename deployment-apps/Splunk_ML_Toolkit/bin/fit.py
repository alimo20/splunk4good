#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import sys
import os

import cexc
import exec_anaconda

try:
    exec_anaconda.exec_anaconda()
except Exception as e:
    cexc.abort(e)
    sys.exit(1)

import cexc
import conf
from cStringIO import StringIO
from util.param_util import is_truthy, parse_args, convert_params
from util import command_util

from chunked_controller import ChunkedController
from cexc import BaseChunkHandler

logger = cexc.get_logger('fit')
messages = cexc.get_messages_logger()


class FitCommand(cexc.BaseChunkHandler):
    """FitCommand uses ChunkedController & one of two processors to fit models.

    The FitCommand can use either the FitBatchProcessor or the FitPartialProcessor,
    which is chosen based on the presence of the partial_fit parameter.
    """

    @staticmethod
    def handle_arguments(getinfo):
        """Take the getinfo metadata and return controller_options.

        Args:
            dict: getinfo metadata from first chunk
        Returns:
            dict: options to be passed to controller
        """
        if len(getinfo['searchinfo']['raw_args']) == 0:
            raise RuntimeError('First argument must be an "algorithm"')

        raw_options = parse_args(getinfo['searchinfo']['raw_args'][1:])
        controller_options, partial_fit = FitCommand.handle_raw_options(raw_options)
        controller_options['algo_name'] = getinfo['searchinfo']['args'][0]
        return controller_options, partial_fit

    @staticmethod
    def handle_raw_options(controller_options):
        """Load command specific options.

        Args:
            dict: options from handle_arguments
        Returns:
            tuple: a dict of controller options and secondly a boolean flag
                for partial fit
        """
        controller_options['processor'] = 'FitBatchProcessor'
        partial_fit = False

        if 'params' in controller_options:
            try:
                fit_params = convert_params(
                    controller_options['params'], ignore_extra=True,
                    bools=['apply', 'partial_fit'])
            except ValueError as e:
                raise RuntimeError(str(e))

            if 'apply' in fit_params:
                controller_options['apply'] = fit_params['apply']
                del controller_options['params']['apply']

            if 'partial_fit' in fit_params:
                partial_fit = fit_params['partial_fit']
                del controller_options['params']['partial_fit']

        if partial_fit:
            controller_options['processor'] = 'FitPartialProcessor'

        return controller_options, partial_fit

    def setup(self):
        """Get options, start controller & watchdog, return command type.

        Returns:
            dict: get info response (command type) and required fields
        """
        self.controller_options, self.partial_fit = self.handle_arguments(self.getinfo)
        self.controller = ChunkedController(self.getinfo, self.controller_options)

        self.watchdog = command_util.get_watchdog(
            self.controller.resource_limits['max_fit_time'],
            self.controller.resource_limits['max_memory_usage_mb'],
            os.path.join(self.getinfo['searchinfo']['dispatch_dir'], 'finalize'))

        required_fields = self.controller.get_required_fields()
        return {'type': 'events', 'required_fields': required_fields}

    def get_output_body(self):
        """Collect output body from controller.

        Returns:
            string: body
        """
        return self.controller.output_results()

    def handler(self, metadata, body):
        """Main handler we override from BaseChunkHandler.

        Returns:
            tuple: metadata, body
        """
        if command_util.is_getinfo_chunk(metadata):
            return self.setup()

        if self.getinfo.get('preview', False):
            logger.debug('Not running in preview.')
            return {'finished': True}

        if not self.watchdog.started:
            self.watchdog.start()

        finished_flag = metadata.get('finished', False)

        self.controller.load_data(body)

        # Partial fit should *always* execute on every chunk.
        # Non partial fit will execute on the last chunk.
        if self.partial_fit or finished_flag:
            self.controller.execute()
            output_body = self.get_output_body()
        else:
            output_body = None

        if finished_flag:
            self.controller.finalize()
            # Gracefully terminate watchdog
            if self.watchdog.started:
                self.watchdog.join()

        # Our final farewell
        self.log_performance_timers()
        return ({'finished': finished_flag}, output_body)

    def log_performance_timers(self):
        logger.debug(
            "command=fit, read_time=%f, handle_time=%f, write_time=%f, csv_parse_time=%f, csv_render_time=%f" % (
                self._read_time, self._handle_time, self._write_time,
                self.controller._csv_parse_time, self.controller._csv_render_time
            ))


if __name__ == "__main__":
    logger.debug("Starting fit.py.")
    do_profile = is_truthy(conf.get_mlspl_prop('profile', 'default', 'n'))

    if do_profile:
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()

    FitCommand(handler_data=BaseChunkHandler.DATA_RAW).run()

    if do_profile:
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        ps = pstats.Stats(pr, stream=s).sort_stats('time')
        ps.print_stats(10)
        logger.info("PROFILE: %s", s.getvalue())

    logger.debug("Exiting gracefully. Byee!!")
