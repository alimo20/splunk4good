#!/usr/bin/env python
# hobbes3

# Rename this file to settings.py to be in effect.

THREADS = 4
SPLUNK_HOME = "/opt/splunk"
SLEEP = 0.3

SPLUNK_USERNAME = "admin"
SPLUNK_PASSWORD = "arresisH3_splunk"

# 1 MB = 1 * 1024 * 1024
LOG_ROTATION_BYTES = 25 * 1024 * 1024
LOG_ROTATION_LIMIT = 100
LOG_PATH = "/mnt/data/mass_index.log"

DATA = [
#    {
#        "file_path": "/mnt/data/irs_990/*.xml",
#        "index": "irs_990",
#        "sourcetype": "irs_990"
#    },
    {
        "file_path": "/mnt/data/gdelt/historical/*.mentions.tsv",
        "index": "gdelt",
        "sourcetype": "gdelt_mention"
    },
    {
        "file_path": "/mnt/data/gdelt/historical/*.export.tsv",
        "index": "gdelt",
        "sourcetype": "gdelt_event"
    },
]

#DATA = [
#    {
#        "file_path": "/mnt/data/samples/irs_990/*.xml",
#        "index": "main",
#        "sourcetype": "irs_990"
#    },
#    {
#        "file_path": "/mnt/data/samples/gdelt/*.mentions.tsv",
#        "index": "main",
#        "sourcetype": "gdelt_mention"
#    },
#    {
#        "file_path": "/mnt/data/samples/gdelt/*.export.tsv",
#        "index": "main",
#        "sourcetype": "gdelt_event"
#    },
#]
