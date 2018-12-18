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
from multiprocessing import RawValue, Lock

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

def retry(retries, error_msg):
    sleep_sec = RETRY_SLEEP[retries]

    logger.error(error_msg.replace("_SEC_", str(sleep_sec)))

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
                logger.error("{}: Error 404 not found! Skipping...".format(url))
                return False

            error_msg = "{}: Error {}. Sleeping for _SEC_ seconds(s)".format(url, e.code)
            retries = retry(retries, error_msg)
            pass
        #except:
        #    error_msg = "Unexpected error: {} - sleeping for _SEC_ seconds(s)".format(sys.exc_info()[0])
        #    retries = retry(retries, error_msg)
        #    pass

def get_csv(data):
    url = data["url"]
    count_skipped = data["count_skipped"]
    count_got = data["count_got"]

    try:
        filename = re.search(r"\/(\d+\..+)\.CSV\.zip$", url).group(1)
    except AttributeError:
        count_skipped.increment()
        logger.error("{}: Regex filename extraction failed. Skipping...".format(url))
        return

    tsv_file = Path(DATA_PATH + "/" + filename + ".tsv")

    if tsv_file.is_file():
        count_skipped.increment()
        logger.debug("{}: File already exist. Skipping...".format(filename))
        return

    response = url_open(url)

    if not response:
        count_skipped.increment()
        logger.error("{}: No response. Skipping...".format(filename))
        return

    z = ZipFile(BytesIO(response))

    for name in z.namelist():
        count_got.increment()
        z.extract(name, DATA_PATH)
        os.rename(name, filename + ".tsv")
        logger.info("{}: Extracted and saved as TSV".format(filename))

if __name__ == "__main__":
    start_time = time.time()

    option = "latest"

    if len(sys.argv) == 2:
        option = sys.argv[1]

    if option == "full":
        URL_ENGLISH = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        URL_TRANSLINGUAL = "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"
    elif option == "latest":
        URL_ENGLISH = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        URL_TRANSLINGUAL = "http://data.gdeltproject.org/gdeltv2/lastupdate-translation.txt"
    else:
        exit("Only valid arguments are \"full\" and \"latest\". Defaults to \"latest\".") 

    setting_file = Path(os.path.dirname(os.path.realpath(__file__)) + "/settings.py")

    if not setting_file.is_file():
        sys.exit("The file settings.py doesn't exist. Please rename settings.py.template to settings.py.")

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=LOG_ROTATION_BYTES, backupCount=LOG_ROTATION_LIMIT)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] (%(threadName)-10s) %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    print("Log file at {}".format(LOG_PATH))

    logger.info("START OF SCRIPT.")
    logger.debug("THREADS={} DATA_PATH={}".format(THREADS, DATA_PATH))

    os.chdir(DATA_PATH)

    # Before retrying first wait 1 second, then another 1, then another 1, then every 30 seconds.
    print("Getting {}".format(URL_ENGLISH))
    text_english = url_open(URL_ENGLISH)
    print("Getting {}".format(URL_TRANSLINGUAL))
    text_translingual = url_open(URL_TRANSLINGUAL)

    urls = []

    urls.extend(text_english.decode("utf-8").splitlines())
    urls.extend(text_translingual.decode("utf-8").splitlines())

    urls = [v.split(" ")[2] for v in urls if re.search(r"(?:export|mentions)\.CSV\.zip$", v)]
    
    data = []

    count_skipped = Counter(0)
    count_got = Counter(0)

    for url in urls:
        data.append({
            "url": url,
            "count_got": count_got,
            "count_skipped": count_skipped
        })

    if TEST_LIMIT != 0:
        del data[TEST_LIMIT:]

    count_total = len(data)

    logger.info("{} URLs to get".format(count_total))

    # https://stackoverflow.com/a/40133278/1150923
    pool = Pool(THREADS)

    for _ in tqdm(pool.imap_unordered(get_csv, data), total=count_total):
        pass

    pool.close()
    pool.join()

    logger.info("Total: {}. Got: {}. Skipped: {}.".format(count_total, count_got.value, count_skipped.value))
    logger.info("Done. Total elapsed seconds: {}".format(time.time() - start_time))
