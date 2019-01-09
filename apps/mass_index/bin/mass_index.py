#!/usr/bin/env python
# hobbes3
# MAKE SURE YOU AUTHENTICATE TO SPLUNK CLI BEFORE RUNNING THIS SCRIPT
# Simply try "splunk login"

import os
import time
import sys
import glob
import subprocess
import pexpect
import logging
import logging.handlers
from tqdm import tqdm
from pathlib import Path
from multiprocessing.dummy import Pool
from multiprocessing import RawValue, Lock
from shutil import copy

from settings import *

class Counter(object):
    def __init__(self, initval=0):
        self.val = RawValue('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value

def index_file(data):
    file_path = data["file_path"]
    #index = data["index"]
    #sourcetype = data["sourcetype"]
    local_counter = 0
    incrementer = 100
    file_paths = sorted(glob.glob(file_path))
    count_total = len(file_paths)	
    logger.debug("{} files to index.".format(count_total))
    	
    #iterate through file paths, moving INCREMENTER files at a time to sinkhole
#    try:
    while count_total > local_counter:
    	for f in file_paths[local_counter:local_counter+incrementer]:
        	logger.debug('Processing filename:{}'.format(f))
        	copy(f, "/opt/splunk/etc/apps/mass_index/data")
        	sinkHoleFiles = len(glob.glob("/opt/splunk/etc/apps/mass_index/data/*"))
        	while sinkHoleFiles > incrementer:
                	sinkHoleFiles = len(glob.glob("/opt/splunk/etc/apps/mass_index/data/*"))
                	logger.debug('Sleeping... number of files in sinkhole{} '.format(sinkHoleFiles))
                	time.sleep(0.5)
    	local_counter = local_counter + incrementer

if __name__ == "__main__":
    start_time = time.time()

    setting_file = Path(os.path.dirname(os.path.realpath(__file__)) + "/settings.py")
    
    if not setting_file.is_file():
        sys.exit("The file settings.py doesn't exist. Please rename settings.py.template to settings.py.")
    
    #set logging stuff
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=LOG_ROTATION_BYTES, backupCount=LOG_ROTATION_LIMIT)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] (%(threadName)-10s) %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    print("Log file at {}".format(LOG_PATH))

    logger.info("START OF SCRIPT.")
    logger.debug("THREADS={} SLEEP={} SPLUNK_HOME={}".format(THREADS, SLEEP, SPLUNK_HOME))
    logger.debug("DATA length: {}".format(len(DATA)))

    data = []

    count_success = Counter(0)
    count_failure = Counter(0)
    count_total = 0
    local_counter = 0
    incrementer = 10

    #glob up file paths and sort
    for i, d in enumerate(DATA):
    	file_path = d["file_path"]
    	logger.debug("DATA #{}: file_path={}".format(i,file_path))
        print("DATA #{}: file_path={}".format(i,file_path))
    	index_file(DATA[i])

