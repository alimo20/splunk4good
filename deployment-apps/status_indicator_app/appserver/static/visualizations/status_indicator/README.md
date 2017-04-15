# Status Indicator

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0/StatusIndicator/StatusIndicatorIntro

## Sample Searches

```
| stats count
```

```
| stats count | eval icon = "warning" | eval color = "#FF00FF"
```

## Data Format
value [, icon, color]