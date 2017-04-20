# Parallel Coordinates Visualization

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0.0/ParallelCoordinates/ParallelCoordIntro

## Sample Searches

```
| inputlookup nutrients.csv | head 1500 | table group calories "protein (g)" "water (g)"
```