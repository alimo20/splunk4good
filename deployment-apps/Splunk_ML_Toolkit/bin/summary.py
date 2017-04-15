#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import sys

import cexc
import exec_anaconda

try:
    exec_anaconda.exec_anaconda()
except Exception as e:
    cexc.abort(e)
    sys.exit(1)

from util import param_util, command_util

from chunked_controller import ChunkedController
from cexc import BaseChunkHandler

logger = cexc.get_logger('summary')
messages = cexc.get_messages_logger()


class SummaryCommand(BaseChunkHandler):
    """Summary command gets model summaries from ML-SPL models."""

    @staticmethod
    def handle_arguments(getinfo):
        """Catch invalid argument and return controller options.

        Args:
            dict: getinfo metadata
        Return:
            dict: controller options
        """
        if len(getinfo['searchinfo']['args']) == 0:
            raise RuntimeError('First argument must be a saved model')

        controller_options = param_util.parse_args(getinfo['searchinfo']['raw_args'][1:])
        controller_options['model_name'] = getinfo['searchinfo']['args'][0]
        controller_options['processor'] = 'SummaryProcessor'
        return controller_options

    def setup(self):
        """Handle args, start controller, and return command type.

        Returns:
            dict: getinfo response
        """
        self.controller_options = self.handle_arguments(self.getinfo)
        self.controller = ChunkedController(self.getinfo, self.controller_options)
        return {'type': 'reporting', 'generating': True}

    def handler(self, metadata, body):
        """Main handler we override from BaseChunkHandler."""
        if command_util.is_getinfo_chunk(metadata):
            return self.setup()

        self.controller.execute()
        body = self.controller.output_results()

        return ({'finished': True}, body)


if __name__ == "__main__":
    logger.debug("Starting summary.py.")
    SummaryCommand(handler_data=BaseChunkHandler.DATA_RAW).run()
    logger.debug("Exiting gracefully. Byee!!")
