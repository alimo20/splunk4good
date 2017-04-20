# Horizon Chart

Documentation:
http://docs.splunk.com/Documentation/CustomViz/1.0/HorizonChart/HorizonChartIntro

## Sample Searches

```
| inputlookup stocks.csv | eval _time = strptime(date, "%Y-%m-%d")| timechart span=1d  latest(open) by ticker_symbol | filldown
```