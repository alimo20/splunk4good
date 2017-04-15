#!/usr/bin/env python

import csv
import errno
import os
import re
import json
import traceback

import lookup_update_notify
import conf
import cexc
from util.base_util import is_valid_identifier

logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()

model_dir = os.path.join('..', 'lookups')


def load_model(model_name, skip_model_obj=False, model_dir=model_dir, tmp=False):
    with open_model_file(model_name, model_dir=model_dir, tmp=tmp) as f:
        model_reader = csv.DictReader(f)
        csv.field_size_limit(2 ** 30)
        model_data = model_reader.next()

        algo_name = model_data['algo']
        model_options = json.loads(model_data['options'])

        if skip_model_obj:
            model_obj = None
        else:
            from base import BaseMixin
            assert (is_valid_identifier(algo_name))
            algos = __import__("algos", fromlist=["*"])
            algo_module = getattr(algos, algo_name)
            algo_class = getattr(algo_module, algo_name)

            if hasattr(algo_class, 'register_codecs'):
                algo_class.register_codecs()
            model_obj = BaseMixin.decode(model_data['model'])

    return (algo_name, model_obj, model_options)


def save_model(model_name, algo, algo_name, options, max_size=None, model_dir=model_dir, tmp=False):
    if algo:
        algo_class = type(algo)
        if hasattr(algo_class, 'register_codecs'):
            algo_class.register_codecs()
        opaque = algo.encode()
    else:
        opaque = ''

    if max_size > 0 and len(opaque) > max_size * 1024 * 1024:
        raise RuntimeError("Model exceeds size limit (%d > %d)" % (
            len(opaque), max_size * 1024 * 1024))

    with open_model_file(model_name, mode='w', model_dir=model_dir, tmp=tmp) as f:
        model_writer = csv.writer(f)

        # TODO: Version attribute
        model_writer.writerow(['algo', 'model', 'options'])
        model_writer.writerow([algo_name, opaque, json.dumps(options)])


def notify_model_update(getinfo, model_name):
    splunkd_uri = getinfo['searchinfo']['splunkd_uri']
    session_key = getinfo['searchinfo']['session_key']
    app = conf.APP_NAME
    user = 'nobody'
    lookup_file = model_name_to_filename(model_name, False)

    lookup_update_notify.lookup_update_notify(
        splunkd_uri, session_key, app, user, lookup_file
    )


def open_model_file(name, mode='r', model_dir=model_dir, tmp=False):
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(model_dir):
            pass
        else:
            # TODO: Log traceback
            raise Exception("Error opening model '%s': %s" % (name, e))

    path = model_name_to_path(name, model_dir=model_dir, tmp=tmp)  # raises if invalid

    # Double check that path is not outside of model_dir.
    if os.path.relpath(path, model_dir)[0:2] == '..':
        raise ValueError('Illegal escape from parent directory "%s": %s' %
                         (model_dir, path))

    return open(path, mode)


def model_name_to_path(name, model_dir=model_dir, tmp=False):
    return os.path.join(model_dir, model_name_to_filename(name, tmp=tmp))


def model_name_to_filename(name, tmp=False):
    assert isinstance(name, basestring)
    assert is_valid_identifier(name), "Invalid model name"

    suffix = '.tmp' if tmp else ''

    return '__mlspl_' + name + '.csv' + suffix


def list_models():
    output = []

    model_re = re.compile('__mlspl_(?P<model_name>[a-zA-Z_][a-zA-Z0-9_]*).csv')
    files = [f for f in os.listdir(model_dir) if model_re.match(f)]
    for _f in files:
        match = model_re.match(_f)
        model_name = match.group('model_name')

        try:
            algo_name, _, options = load_model(model_name, skip_model_obj=True)
            output.append([model_name, algo_name, json.dumps(options)])
        except Exception as e:
            logger.warn(traceback.format_exc())
            messages.warn('listmodels: Failed to load model "%s"', model_name)

    return output


def delete_model(model_name, model_dir=model_dir, tmp=False):
    path = model_name_to_path(model_name, model_dir=model_dir, tmp=tmp)

    # Double check that path is not outside of model_dir.
    if os.path.relpath(path, model_dir)[0:2] == '..':
        raise ValueError('Illegal escape from parent directory "%s": %s' %
                         (model_dir, path))

    os.unlink(path)
