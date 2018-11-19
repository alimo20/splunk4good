# This file will contain possible attribute/value pairs for configuring scoring methods.

# There is an scorings.conf in $SPLUNK_HOME/etc/apps/Splunk_ML_Toolkit/default/. You must restart
# Splunk to enable configurations.

# To learn more about configuration files (including precedence) please see the
# documentation located at
# http://docs.splunk.com/Documentation/Splunk/latest/Admin/Aboutconfigurationfiles

# GLOBAL SETTINGS
# Use the [default] stanza to define any global settings.
#   * You can also define global settings outside of any stanza, at the top of
#     the file.
#   * Each conf file should have at most one default stanza. If there are
#     multiple default stanzas, attributes are combined. In the case of
#     multiple definitions of the same attribute, the last definition in the
#     file wins.
#   * If an attribute is defined at both the global level and in a specific
#     stanza, the value in the specific stanza takes precedence.

[<STANZA_NAME>]
* Each stanza represents an ML-SPL scoring method; the scoring method name is the stanza name.
* Set the following attributes/values for the command.  Otherwise, Splunk uses the defaults.
* Implementation of the scoring methods should be located in
  $SPLUNK_HOME/etc/apps/Splunk_ML_Toolkit/bin/scorings/.

module = <string>
* The file name of the Python script that contains the scoring method.

class = <string>
* The class name of the scoring method in the Python script.

package = <string>
* Package specifies the location of the directory that contains scoring modules.
