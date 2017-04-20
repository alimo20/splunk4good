#!/usr/bin/env python
# Copyright (C) 2015-2017 Splunk Inc. All Rights Reserved.
import cexc
import models
import pandas as pd
from BaseProcessor import BaseProcessor

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


class ListModelsProcessor(BaseProcessor):
    """The list models processor lists the saved ML-SPL models."""

    @staticmethod
    def list_models():
        """Create the output table of models and options."""
        list_of_models = [dict(zip(('name', 'type', 'options'), m))
                          for m in models.list_models()]

        return pd.DataFrame(list_of_models)

    def process(self):
        """List the saved models."""
        self.df = self.list_models()
