# Location Tracker

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0.0/RealTimeLocation/RealTimeTrackerIntro

## Sample Queries

```
| inputlookup locations.csv | table _time lat lon user
```