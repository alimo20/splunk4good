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

from util import command_util
from chunked_controller import ChunkedController
from cexc import BaseChunkHandler

logger = cexc.get_logger('list')
messages = cexc.get_messages_logger()


class ListModelsCommand(BaseChunkHandler):
    """ListModelsCommand uses the ChunkedController & ListModelsProcessor to list saved models."""

    @staticmethod
    def handle_arguments(getinfo):
        """Check for invalid arguments and get controller_options.

        Args:
            dict: getinfo metadata
        Returns:
            dict: controller options
        """
        if len(getinfo['searchinfo']['args']) > 0:
            raise RuntimeError('Invalid arguments')  # TODO: more descriptive error message

        controller_options = {}
        controller_options['processor'] = 'ListModelsProcessor'
        return controller_options

    def setup(self):
        """Get options, start controller, return command type.

        Returns:
            dict: get info response (command type)
        """
        self.controller_options = self.handle_arguments(self.getinfo)
        self.controller = ChunkedController(self.getinfo, self.controller_options)
        return {'type': 'reporting', 'generating': True}

    def handler(self, metadata, body):
        """Default handler we override from BaseChunkHandler."""
        if command_util.is_getinfo_chunk(metadata):
            return self.setup()

        # Don't run in preview.
        if self.getinfo.get('preview', False):
            logger.debug('Not running in preview')
            return {'finished': True}

        self.controller.execute()
        body = self.controller.output_results()

        # Final farewell
        return ({'finished': True}, body)


if __name__ == "__main__":
    logger.debug("Starting list.py.")
    ListModelsCommand(handler_data=BaseChunkHandler.DATA_RAW).run()
    logger.debug("Exiting gracefully. Byee!!")
