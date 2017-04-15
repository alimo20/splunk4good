# Sankey Diagram

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0/Sankey/SankeyIntro

## Sample Searches
```
index=_internal sourcetype=splunk_web_access NOT uri_path=*/static/* uri_path=*/app/* OR uri_path=*/manager/*   | rex field=referer "https?://.+?/.+?(?<referer_path>/[^\\?]+)"   | rex field=uri_path "/.+?(?<path>/.+)"   | rename referer_path as from path as to | stats count, avg(bytes) by from to
```