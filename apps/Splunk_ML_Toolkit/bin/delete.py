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

logger = cexc.get_logger('delete')
messages = cexc.get_messages_logger()


class DeleteModelCommand(BaseChunkHandler):
    """DeleteModelCommand uses the ChunkedController & DeleteModelProcessor to delete models."""

    @staticmethod
    def handle_arguments(getinfo):
        """Check for invalid argument usage and return controller options.

        Args:
            dict: getinfo metadata
        Returns:
            dict: controller options
        """
        if len(getinfo['searchinfo']['args']) != 1:
            raise RuntimeError('Usage: deletemodel <modelname>')

        controller_options = {}
        controller_options['model_name'] = getinfo['searchinfo']['args'][0]
        controller_options['processor'] = 'DeleteModelProcessor'
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

        self.controller.execute()
        return {'finished': True}


if __name__ == "__main__":
    logger.debug("Starting delete.py.")
    DeleteModelCommand().run()
    logger.debug("Exiting gracefully. Byee!!")
