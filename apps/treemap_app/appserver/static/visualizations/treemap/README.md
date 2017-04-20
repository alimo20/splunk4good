# Treemap

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0.0/TreeMap/TreeMapIntro

## Sample searches

Using a single measure to drive rectangle **size**:

```
| inputlookup splunk_files.csv 
| rex field=path "\./(?<level1>[A-Za-z0-9]*)/(?<level2>[A-Za-z0-9]*)" 
| stats sum(size) as size by level1, level2
```

Using two measures, the first to drive rectangle **size** and the second using a numeric field to drive rectangle **color**:

```
| inputlookup splunk_files.csv 
| rex field=path "\./(?<level1>[A-Za-z0-9]*)/(?<level2>[A-Za-z0-9]*)" 
| stats count, sum(size) as size by level1, level2
```

Using two measures, the first to drive rectangle **size** and the second using a string (category) field to drive rectangle **color**:

```
| inputlookup splunk_files.csv 
| rex field=path "\./(?<level1>[A-Za-z0-9]*)/(?<level2>[A-Za-z0-9]*)"
| rex field=path ".+\.(?<extension>.*)$"
| stats count, sum(size) by level1, level2, extension
```