#!/usr/bin/python

# Written by hobbes3

import paramiko
import socket
import traceback
import json
import sys
import glob
import os
from pipes import quote

print "Symlinking master-apps to apps..."

os.chdir("/opt/splunk/etc/master-apps/")
for app in os.listdir("."):
    if os.path.islink("../apps/" + app):
        os.unlink("../apps/" + app)
    os.symlink("../master-apps/" + app, "../apps/" + app)

hostname = socket.gethostname()

host = "c0m1.splunk4good_prod.splunkcloud.com"
port = 22

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print "Connecting to " + host + "..."
    client.connect(host, port=port)
    
    print "git pull..."
    stdin, stdout, stderr = client.exec_command("cd /opt/splunk/etc/splunk4good/; git pull")
    print stdout.read()
    
    print "git updating submodules..."
    stdin, stdout, stderr = client.exec_command("cd /opt/splunk/etc/splunk4good/; git submodule update --recursive --remote")
    print stdout.read()


    stdin, stdout, stderr = client.exec_command("""
        cd /opt/splunk/etc/splunk4good/master-apps/;
        for app in *;
            do echo "symlinking $app...";
            ln -s -f ../splunk4good/master-apps/$app ../../master-apps/$app;
        done
    """)
    print stdout.read()

    print "Reading splunk4good_logins.json..."
    with open("../splunk4good_logins.json") as json_file:    
        logins = json.load(json_file)

    user     = logins[host]["user"]
    password = logins[host]["password"]

    stdin, stdout, stderr = client.exec_command("/opt/splunk/bin/splunk apply cluster-bundle -auth " + user + ":" + quote(password))
    print stderr.read()
    print stdout.read()

except Exception as e:
    print "*** Caught exception: " + str(e.__class__) + ": " + str(e)
    traceback.print_exc()
    try:
        t.close()
    except:
        pass
    sys.exit(1)

finally:
    print "Closing connection..."
    client.close()
