#!/usr/bin/env python

import os


def get_splunkhome_path():
    return os.path.normpath(os.environ["SPLUNK_HOME"])


def make_splunkhome_path(p):
    return os.path.join(get_splunkhome_path(), *p)


def get_etc_path():
    return os.environ.get(
        'SPLUNK_ETC',
        os.path.join(get_splunkhome_path(), 'etc'))


def get_apps_path():
    return os.path.normpath(os.path.join(get_etc_path(), 'apps'))
