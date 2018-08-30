"""
This module contains utility methods needed by both models.base, models.listmodels and models.deletemodels
"""

import json
import os
import shutil

import cexc
from util.base_util import is_valid_identifier
from util.lookups_util import load_lookup_file_from_disk
from util import rest_url_util
from util.rest_proxy import rest_proxy_from_searchinfo
from util.lookups_util import parse_model_reply, lookup_name_to_path_from_splunk


logger = cexc.get_logger(__name__)
messages = cexc.get_messages_logger()


def get_model_list_by_experiment(rest_proxy, namespace, experiment_id):
    url_params = [
        ('search', '__mlspl__exp_*{}*.csv'.format(experiment_id)),
        ('count', '0')
    ]
    url = rest_url_util.make_get_lookup_url(rest_proxy, namespace=namespace, url_params=url_params)
    reply = rest_proxy.make_rest_call('GET', url)
    content = json.loads(reply.get('content'))
    entries = content.get('entry')
    model_list = []
    for entry in entries:
        model_list.append(entry.get('name'))
    return model_list


def load_algo_options_from_disk(file_path):
    model_data = load_lookup_file_from_disk(file_path)
    algo_name = model_data['algo']
    model_options = json.loads(model_data['options'])

    return algo_name, model_data, model_options


def model_name_to_filename(name, tmp=False):
    assert isinstance(name, basestring)
    assert is_valid_identifier(name), "Invalid model name"
    suffix = '.tmp' if tmp else ''
    return '__mlspl_' + name + '.csv' + suffix


def move_model_file_from_staging(model_filename, searchinfo, namespace, model_filepath):
    rest_proxy = rest_proxy_from_searchinfo(searchinfo)
    url = rest_url_util.make_lookup_url(rest_proxy, namespace=namespace, lookup_file=model_filename)

    payload = {
        'eai:data': model_filepath,
        'output_mode': 'json'
    }

    # try to update the model
    reply = rest_proxy.make_rest_call('POST', url, postargs=payload)

    # if we fail to update the model because it doesn't exist, try to create it instead
    if not reply['success']:
        if reply['error_type'] == 'ResourceNotFound':
            payload['name'] = model_filename
            reply = rest_proxy.make_rest_call('POST', url, postargs=payload)

        # the redundant-looking check is actually necessary because it prevents this logic from triggering if the update fails but the create succceeds
        if not reply['success']:
            try:
                # if the model save fails, clean up the temp model file
                os.unlink(model_filepath)
            # if we somehow fail to clean up the temp model, don't expose the error to the user
            except Exception as e:
                logger.debug(str(e))

            parse_model_reply(reply)


def copy_model_to_staging(src_model_name, dest_model_name, searchinfo, dest_dir_path):
    """
    given a model name and space info, disk copy the model file to a destined directory with a new model name
    Args:
        src_model_name (str): source model name
        dest_model_name (str): destination model name, the name could be different
        searchinfo (dict): searchinfo of the model owner
        dest_dir_path (str): destination path 

    Returns:
        (filename, filepath) (tuple): the file name and file path in staging directory
    """
    src_model_filename = model_name_to_filename(src_model_name)
    src_model_filepath = lookup_name_to_path_from_splunk(src_model_name, src_model_filename, searchinfo, namespace='user', lookup_type='model')

    dest_model_filename = model_name_to_filename(dest_model_name)
    dest_model_filepath = os.path.join(dest_dir_path, dest_model_filename)
    try:
        shutil.copy2(src_model_filepath, dest_model_filepath)
    except Exception as e:
        logger.debug(str(e))
        raise Exception('Cannot find experiment draft model')

    return dest_model_filename, dest_model_filepath
