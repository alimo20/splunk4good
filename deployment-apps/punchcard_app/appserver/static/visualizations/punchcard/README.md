# Punchcard

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0.0/Punchcard/PunchcardIntro

## Sample Searches

```
index=_internal | head 100000 | stats count by date_hour sourcetype
```

```
index=_internal | head 100000 | stats count by date_minute sourcetype
```

```
index=_internal | head 100000 | stats count, avg(date_minute) by date_minute sourcetype
```