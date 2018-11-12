#!/usr/bin/env python
# hobbes3

import os
import re
import time
import sys
import logging
import logging.handlers
import urllib.request
import urllib.error
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
from multiprocessing.dummy import Pool

from settings import *

def retry(retries, error_msg):
    sleep_sec = RETRY_SLEEP[retries]

    logger.debug(error_msg.replace("_SEC_", str(sleep_sec)))

    time.sleep(sleep_sec)

    if retries >= len(RETRY_SLEEP) - 1:
        retries = len(RETRY_SLEEP) - 1
    else:
        retries += 1

    return retries

def url_open(url):
    retries = 0

    while True:
        logger.info(url)

        try:
            response = urllib.request.urlopen(url)

            return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug("{}: Error 404 not found! Skipping...".format(url))
                return False

            error_msg = "{}: Error {}. Sleeping for _SEC_ seconds(s)".format(url, e.code)
            retries = retry(retries, error_msg)
            pass
        #except:
        #    error_msg = "Unexpected error: {} - sleeping for _SEC_ seconds(s)".format(sys.exc_info()[0])
        #    retries = retry(retries, error_msg)
        #    pass

def get_csv(url):
    try:
        filename = re.search(r"\/(\d+\.\w+)\.CSV\.zip$", url).group(1)
    except AttributeError:
        logger.debug("{}: Regex filename extraction failed. Continuing...".format(url))
        pass

    tsv_file = Path(SPLUNK_DATA_PATH + "/" + filename + ".tsv")

    if tsv_file.is_file():
        logger.debug("{}: File already exist. Skipping...".format(filename))
        return

    response = url_open(url)

    if not response:
        return

    z = ZipFile(BytesIO(response))

    for name in z.namelist():
        z.extract(name, SPLUNK_DATA_PATH)
        os.rename(name, filename + ".tsv")
        logger.info("{}: Extracted and saved as TSV".format(filename))

if __name__ == "__main__":
    start_time = time.time()

    URL_ENGLISH = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    URL_TRANSLINGUAL = "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"

    setting_file = Path(os.path.dirname(os.path.realpath(__file__)) + "/settings.py")

    if not setting_file.is_file():
        sys.exit("The file settings.py doesn't exist. Please rename settings.py.template to settings.py.")

    logger = logging.getLogger('logger_debug')
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(SPLUNK_LOG_PATH + "/create_data.log", maxBytes=LOG_ROTATION_BYTES, backupCount=1000)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] (%(threadName)-10s) %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    logger.info("START OF SCRIPT.")

    os.chdir(SPLUNK_DATA_PATH)

    # Before retrying first wait 1 second, then another 1, then another 1, then every 30 seconds.
    print("Getting {}".format(URL_ENGLISH))
    text_english = url_open(URL_ENGLISH)
    print("Getting {}".format(URL_TRANSLINGUAL))
    text_translingual = url_open(URL_TRANSLINGUAL)

    urls = []

    urls.extend(text_english.decode("utf-8").splitlines())
    #urls.extend(text_translingual.decode("utf-8").splitlines())

    urls = [v.split(" ")[2] for v in urls if re.search(r"(?:export|mentions)\.CSV\.zip$", v)]

    #urls = [
    #    "http://data.gdeltproject.org/gdeltv2/20150218230000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218230000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218231500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218231500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218233000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218233000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218234500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150218234500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219000000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219000000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219001500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219001500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219003000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219003000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219004500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219004500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219010000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219010000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219011500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219011500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219013000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219013000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219014500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219014500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219020000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219020000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219021500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219021500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219023000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219023000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219024500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219024500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219030000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219030000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219031500.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219031500.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219033000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219033000.mentions.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219040000.export.CSV.zip",
    #    "http://data.gdeltproject.org/gdeltv2/20150219040000.mentions.CSV.zip"
    #]

    if TEST_LIMIT != 0:
        del urls[TEST_LIMIT:]

    logger.debug("{} URLs to get".format(len(urls)))

    # https://stackoverflow.com/a/40133278/1150923
    pool = Pool(THREADS)

    for _ in tqdm(pool.imap_unordered(get_csv, urls), total=len(urls)):
        pass

    pool.close()
    pool.join()

    logger.debug("Done. Total elapsed seconds: {}".format(time.time() - start_time))
