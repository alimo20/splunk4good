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

import conf
from util.param_util import parse_args, is_truthy
from util import command_util

from chunked_controller import ChunkedController
from cexc import BaseChunkHandler

logger = cexc.get_logger('apply')
messages = cexc.get_messages_logger()


class ApplyCommand(BaseChunkHandler):
    """ApplyCommand uses the ChunkedController & ApplyProcessor to make predictions."""

    @staticmethod
    def handle_arguments(getinfo):
        """Take the getinfo metadata and return controller_options.

        Args:
            dict: getinfo metadata
        Returns:
            dict: options to be sent to controller
        """
        if len(getinfo['searchinfo']['args']) == 0:
            raise RuntimeError('First argument must be a saved model.')

        raw_options = parse_args(getinfo['searchinfo']['raw_args'][1:])
        controller_options = ApplyCommand.handle_raw_options(raw_options)
        controller_options['model_name'] = getinfo['searchinfo']['args'][0]
        return controller_options

    @staticmethod
    def handle_raw_options(raw_options):
        """Load command specific options.

        Args:
            dict: raw options
        Returns:
            dict: modified raw_options
        """
        raw_options['processor'] = 'ApplyProcessor'

        if 'args' in raw_options:
            raise RuntimeError('Apply does not accept positional arguments.')
        return raw_options

    def setup(self):
        """Parse search string, choose processor, initialize controller.

        Returns:
            dict: get info response (command type) and required fields. This
                response will be sent back to the CEXC process on the getinfo
                exchange (first chunk) to establish our execution type and
                required fields.
        """
        self.controller_options = self.handle_arguments(self.getinfo)
        self.controller = ChunkedController(self.getinfo, self.controller_options)

        self.watchdog = command_util.get_watchdog(
            time_limit=-1,
            memory_limit=self.controller.resource_limits['max_memory_usage_mb']
        )

        streaming_apply = is_truthy(conf.get_mlspl_prop('streaming_apply', default='f'))
        exec_type = 'streaming' if streaming_apply else 'stateful'

        required_fields = self.controller.get_required_fields()
        return {'type': exec_type, 'required_fields': required_fields}

    def handler(self, metadata, body):
        """Main handler we override from BaseChunkHandler.

        Handles the reading and writing of data to the CEXC process, and
        finishes negotiation of the termination of the process.

        Args:
            dict: metadata information
            str: body
        Returns:
            tuple: metadata, body
        """
        # Get info exchange an initialize controller, processor, algorithm
        if command_util.is_getinfo_chunk(metadata):
            return self.setup()

        finished_flag = metadata.get('finished', False)

        if not self.watchdog.started:
            self.watchdog.start()

        # Skip to next chunk if this chunk is empty
        if len(body) == 0:
            return {}

        # Load data, execute and collect results.
        self.controller.load_data(body)
        self.controller.execute()
        output_body = self.controller.output_results()

        if finished_flag:
            # Gracefully terminate watchdog
            if self.watchdog.started:
                self.watchdog.join()

        # Our final farewell
        return ({'finished': finished_flag}, output_body)


if __name__ == "__main__":
    logger.debug("Starting apply.py.")
    ApplyCommand(handler_data=BaseChunkHandler.DATA_RAW).run()
    logger.debug("Exiting gracefully. Byee!!")
