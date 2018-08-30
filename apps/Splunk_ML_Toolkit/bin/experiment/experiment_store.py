import os
import json
import httplib
import uuid
import logging

from util import rest_url_util
from models import deletemodels
from exec_anaconda import get_staging_area_path
from util.searchinfo_util import searchinfo_from_request
from util.models_util import (
    move_model_file_from_staging,
    copy_model_to_staging,
    get_model_list_by_experiment,
)
from util.rest_proxy import rest_proxy_from_searchinfo
from models.deletemodels import delete_model_with_splunk_rest
from util.experiment_util import get_experiment_draft_model_name
from experiment.experiment_validation import validate_experiment_form_args
from rest.proxy import SplunkRestEndpointProxy, SplunkRestProxyException

import cexc
logger = cexc.get_logger(__name__)


class ExperimentStore(SplunkRestEndpointProxy):
    """
    API for experiment's conf storage backend
    """

    URL_PARTS_PREFIX = ['configs', 'conf-experiments']
    JSON_OUTPUT_FLAG = ('output_mode', 'json')

    _with_admin_token = False
    _with_raw_result = True
    _promote_model = False

    @property
    def with_admin_token(self):
        return self._with_admin_token

    @property
    def with_raw_result(self):
        return self._with_raw_result

    def _convert_url_parts(self, url_parts):
        """
        - Mandatory overridden
        - see SplunkRestEndpointProxy._convert_url_parts()
        """

        return self.URL_PARTS_PREFIX + url_parts

    def _transform_request_options(self, options, url_parts, request):
        """
        - Overridden from SplunkRestEndpointProxy
        - Handling experiment specific modification/handling of the request before
        sending the request to conf endpoint
        - See RestProxy.make_rest_call() for a list of available items for `options`
        
        Args:
            options (dict): default options constructed by the request method (get, post, delete)
            url_parts (list): a list of url parts without /mltk/experiments
            request (dict): the original request from the rest call to /mltk/experiments/*
        
        Raises:
            SplunkRestProxyException: some error occurred during this process
        
        Returns:
            options: parameters needed by RestProxy.make_rest_call() stored as dictionary
        """

        # for GET/DELETE request, we just want to append output=json to the existing getargs
        if options['method'] == 'GET' or options['method'] == 'DELETE':
            options['getargs'] = dict(options.get('getargs', []) + [self.JSON_OUTPUT_FLAG])

        if options['method'] == 'DELETE':
            self._delete_models(request, url_parts)
        
        # for POST request, we do validation before proxying the request
        if options['method'] == 'POST':
            postargs, blockedargs = self._split_tuple_list(request.get('form', []), ["promoteModel"])
            try:
                validate_experiment_form_args(postargs)
            except Exception as e:
                logger.error(str(e))
                raise SplunkRestProxyException('Can not validate experiment', logging.ERROR, httplib.BAD_REQUEST)

            postargs['output_mode'] = 'json'

            if blockedargs.get('promoteModel', None):
                self._promote_model = True

            if len(url_parts) == 0:
                # this is a create POST
                experiment_uuid = str(uuid.uuid4()).replace('-', '') # removing '-' due to model name constraints
                postargs['name'] = experiment_uuid

            options['postargs'] = postargs

        return options

    def _promote_draft_models(self, reply, request):
        """
        This function serves as a side effect of experiment POST.  It will copy the draft model of each experiment 
        search stage to a "production" name of the model.
        Args:
            reply (dict): the successful reply to the POST request of experiment/{guid}
            request (dict): the POST request to experiments/{guid}
        Returns:
            None
        """
        content = json.loads(reply.get('content'))
        entries = content.get('entry')
        searchinfo = searchinfo_from_request(request)
        for entry in entries:
            ss_json = entry.get('content', {}).get('searchStages')
            if ss_json:
                search_stages = json.loads(ss_json)
            else:
                search_stages = []
            for search_stage in search_stages:
                model_name = search_stage.get('modelName')
                if model_name is not None:
                    draft_model_name = get_experiment_draft_model_name(model_name)
                    staging_model_filename, staging_model_filepath =copy_model_to_staging(draft_model_name,
                                                                                          model_name,
                                                                                          searchinfo,
                                                                                          dest_dir_path=get_staging_area_path())

                    if os.access(staging_model_filepath, os.R_OK):
                        move_model_file_from_staging(staging_model_filename, searchinfo, namespace='user', model_filepath=staging_model_filepath)
                    else:
                        raise Exception('The temp model file %s is missing or permission denied' % staging_model_filename)
                    try:
                        deletemodels.delete_model_from_disk(model_name, model_dir=get_staging_area_path(), tmp=True)
                    except Exception as e:
                        cexc.log_traceback()
                        logger.warn('Exception while deleting tmp model "%s": %s', model_name, e)

    def _delete_models(self, request, url_parts):
        if len(url_parts) == 1:
            try:
                searchinfo = searchinfo_from_request(request)
                rest_proxy = rest_proxy_from_searchinfo(searchinfo)
                model_list = get_model_list_by_experiment(rest_proxy, namespace='user', experiment_id=url_parts[0])
                for model_name in model_list:
                    url = rest_url_util.make_get_lookup_url(rest_proxy, namespace='user', lookup_file=model_name)
                    reply = rest_proxy.make_rest_call('DELETE', url)
            except Exception as e:
                cexc.log_traceback()
                pass

    def _handle_reply(self, reply, request, url_parts, method):
        """
        - Overridden from SplunkRestEndpointProxy
        - Replace '/configs/conf-experiments' in the reply with '/mltk/experiments'
        
        Args:
            reply (dict): the reply we got from '/configs/conf-experiments'
            url_parts (list): a list of url parts without /mltk/experiments
            method (string): original request's method
        
        Returns:
            reply: reply from input after the filtering
        """

        def deproxy(string):
            # replace '/configs/conf-experiments' with '/mltk/experiments'
            return string.replace('/%s' % '/'.join(self.URL_PARTS_PREFIX), '/mltk/experiments')

        content = json.loads(reply.get('content'))

        if content.get('origin'):
            content['origin'] = deproxy(content['origin'])

        if content.get('links'):
            for key, value in content['links'].iteritems():
                content['links'][key] = deproxy(value)

        if content.get('entry'):
            entry = content['entry']
            for item in entry:
                item['id'] = deproxy(item['id'])
                for key, value in item['links'].iteritems():
                    item['links'][key] = deproxy(value)

        # promote the draft model to production.
        if self._promote_model and method == 'POST' and reply.get('status') == httplib.OK:
            self._promote_draft_models(reply, request)

        return {
            'status': reply.get('status', httplib.OK),
            'payload': json.dumps(content)
        }
