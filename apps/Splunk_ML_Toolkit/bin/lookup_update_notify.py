#!/usr/bin/env python
# Simple script to bounce the lookup-update-notify REST endpoint.
# Needed to trigger model replication in SHC environments.
#
# This needs to launch a subprocess for the simple reason that the
# interpreter that ML-SPL runs under (Splunk_SA_Scientific_Python)
# doesn't have OpenSSL. We marshall the required Splunk auth token
# through stdin to avoid leaks via environment variables/command line
# arguments/etc.

import subprocess
import json
import os
import sys

import cexc

logger = cexc.get_logger(__name__)

endpoint = '/services/replication/configuration/lookup-update-notify'


def lookup_update_notify(splunkd_uri, session_key, app, user, lookup_file):
    payload = {
        'splunkd_uri': splunkd_uri,
        'session_key': session_key,
        'app': app,
        'user': user,
        'lookup_file': lookup_file
    }

    try:
        python_path = os.path.join(os.environ['SPLUNK_HOME'], 'bin', 'python')
        p = subprocess.Popen([python_path, os.path.abspath(__file__)],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (stdoutdata, stderrdata) = p.communicate(json.dumps(payload))
        p.wait()

        for errline in stderrdata.splitlines():
            logger.debug('> %s', errline)

        if p.returncode != 0:
            raise RuntimeError("Subprocess exited with non-zero error code '%d'" % p.returncode)

        reply = json.loads(stdoutdata)
    except Exception as e:
        logger.warn('lookup_update_notify failure: %s: %s', type(e).__name__, str(e))
        return False

    logger.debug('lookup_update_notify reply: %s', reply)

    return reply['success']


if __name__ == "__main__":
    import splunk
    from splunk import rest

    # Read JSON payload from stdin
    try:
        line = sys.stdin.next()
        payload = json.loads(line)

        splunkd_uri = payload['splunkd_uri']
        session_key = payload['session_key']

        request = {
            'app': payload['app'],
            'user': payload['user'],
            'filename': payload['lookup_file'],
        }

        response, content = rest.simpleRequest(
            splunkd_uri + endpoint, method='POST', postargs=request,
            sessionKey=session_key, raiseAllErrors=False)

        reply = {
            'success': False,
            'response': response,
            'content': content,
        }

        if response.status == 200:
            reply['success'] = True
        elif response.status == 400 and 'ConfRepo' in content:
            reply['success'] = True
    except Exception as e:
        reply = {
            'success': False,
            'content': '%s: %s' % (type(e).__name__, str(e))
        }

    print json.dumps(reply)
