# S4G California Drought

![https://dev.splunk4good.com/en-US/app/s4g-ca_drought/](https://img.shields.io/badge/App%20Status-Stable%20(beta)-orange.svg) ![https://splunkbase.splunk.com/app/1257/](https://img.shields.io/badge/Dependencies-Wunderground%20TA-green.svg)
![https://dev.splunk4good.com/en-US/app/s4g-ca_drought/](https://img.shields.io/badge/Deployment%20Status-Dev-orange.svg)

## UPDATE
`hobbes3` (May 2017): I've made massive edits and cleaned up this app for the Splunk4Good environment. Some parts of the documentation may be no longer relevant.

## Overview

The state of California has been suffering from drought for the past four years. This has led to mandatory rationing and other conservation measures to be adopted in the state.

Agriculture in California has consistently consumed around 80 percent of California’s fresh water, even as it accounts for only 2 percent of the state’s gross domestic product.

Yet majority of the conservation efforts focus only on reducing water consumption in urban areas instead of focusing arcane water rights laws.

A shift to more efficient irrigation methods could reduce agricultural water use by 22 percent, an amount equivalent to all the surface water Central Valley farmers lacked because of drought last year, according to an analysis that Cooley of the Pacific Institute co-authored with Robert Wilkinson, a professor at the University of California Santa Barbara, and Kate Poole, a senior attorney at the Natural Resources Defense Council. 

Water is still priced more cheaply than it should be, which encourages over-consumption. One reason is that much of the state’s water is provided by federal and state agencies at prices that taxpayers subsidize. A second factor that encourages waste is the “use it or lose it” feature in California’s arcane system of water rights. Under current rules, if a property owner does not use all the water to which he is legally entitled, he relinquishes his future rights to the unused water, which may then get allocated to the next farmer in line.

Arizona  and California are the only two states that count and regulate groundwater and surface water as separate source despite the overlap/ interconnection between surface water and groundwater being proven ages ago.  Drawing groundwater from near a stream can suck that stream dry. In turn, using all the water in streams and rivers lessens their ability to replenish the aquifers beneath them.

California was the only state in the arid West that set no limits on how much groundwater a property owner could extract from a private well. This led to farmers digging deeper wells. California is the only state where Well completion records are not public.

Last year a bill was passed that focused on regulating groundwater withdrawals.This bill contained a provision that explicitly prohibits California state regulators from addressing the interconnection between groundwater and surface water in local water plans until 2025 as doing so threatened to derail the legislation entirely, triggering fierce opposition from the Agricultural Council of California, the California Chamber of Commerce and other industry groups.

## App Installation

This app has been built for Splunk 6.3+ (choropleth maps)

1) Intall this package into $SPLUNK_HOME> etc > apps

2) Most data within this app is self contained (found in [/datafiles](/datafiles)). All you need to do is go to inputs.conf / Splunk GUI and enable the data sources for this app. For API inputs you will need to add a [Wunderground key which you can obtain at no cost](http://www.wunderground.com/weather/api).

Notes: The app will create 2 indexes: "s4g-cad_prod" and "s4g-cad_test". All bundled data is configured to the "cs4g-cad_prod" and inputs are enabled by default.

## The Data

####[OECD Worldwide Water Use](http://www.oecd-ilibrary.org/sites/factbook-2014-en/tables/table-153.html?contentType=&itemId=%2fcontent%2ftable%2ffactbook-2014-table153-en&mimeType=text%2fhtml&containerItemId=%2fcontent%2fserial%2f18147364&accessItemIds=%2fcontent%2fchapter%2ffactbook-2014-68-en)
Annual water consumption per capita in selected countries (in cubic meters).

|         |            |
| ------------- |-------------|
| Data download      | [web](http://www.oecd-ilibrary.org/sites/factbook-2014-en/tables/table-153.html?contentType=&itemId=%2fcontent%2ftable%2ffactbook-2014-table153-en&mimeType=text%2fhtml&containerItemId=%2fcontent%2fserial%2f18147364&accessItemIds=%2fcontent%2fchapter%2ffactbook-2014-68-en) |
| Parse before indexing | . |
| Sourcetypes | cad_oecd_water_use |
| Dashboards | index |
| Lookups | . |
| Datamodel | cad_oecd_water_use |
| New Data Updated | . |
| Current Data Available | 2011 |
| Data Granularity | year |
| Contacts | .  |

_____

####[State Water Resources Control Board](http://www.waterboards.ca.gov/water_issues/programs/conservation_portal/conservation_reporting.shtml)

This dataset is made available by State Water Resources Control Board. Water suppliers having more than 3000 connections are required to report on their conservation efforts starting June 2014. Water usage in the year 2013 is considered as the baseline and the Conservation Goal is to reduce usage by 25% as compared to 2013.

|         |            |
| ------------- |-------------|
| Data download | [web](http://www.waterboards.ca.gov/water_issues/programs/conservation_portal/conservation_reporting.shtml) |
| Parse before indexing | Yes - [Scripted UWS formatter.py](bin/scripts/cad_uws_reports/UWS formatter.py) [(Instructions)](datafiles/data_formatting_process.docx) |
| Sourcetypes | cad_uws_reports |
| Dashboards | urban_water_supplier_reports, urban_water_supplier_explore |
| Lookups | supplier_tiers, water_supplier_regions     |
| Datamodel | cad_uws_reports |
| New Data Updated | monthly |
| Current Data Available | June 2014 - Sept 2015 |
| Data Granularity | day  |
| Contacts | .  |

_____

####[USGS Groundwater Data California Department of Water Resources - Groundwater](http://waterdata.usgs.gov/nwis/gw)

Groundwater levels in the state are monitored by jointly by USGS and CaDWR.

|         |            |
| ------------- |-------------|
| Data download  | [web](http://waterservices.usgs.gov) |
| Parse before indexing | yes - [GroundwaterData_retrieval.py](bin/scripts/cad_usgs_groundwater/GroundwaterData_retrieval.py) [(instructions)](datafiles/data_formatting_process.docx) |
| Sourcetypes | cad_usgs_groundwater |
| Dashboards | reservoir_and_groundwater_levels     |
| Lookups | .     |
| Datamodel | .     |
| New Data Updated | Monthly     |
| Current Data Available | January 2010 - June 2015     |
| Data Granularity | day     |
| Contacts | .     |

_____

####[DWR Groundwater Data]

|         |            |
| ------------- |-------------|
| Data download | email |
| Parse before indexing | yes - [(instructions)](datafiles/data_formatting_process.docx) |
| Sourcetypes | cad_dwr_groundwater |
| Dashboards | reservoir_and_groundwater_levels |
| Lookups | DWR-GST |
| Datamodel | cad_dwr_groundwater |
| New Data Updated | Every 3-6 months (or whenever they oblige to process the request) |
| Current Data Available | Jan 2001 - Jun 2015     |
| Data Granularity | day  |
| Contacts | eric.senter@water.ca.gov |

_____

####[California Departmeant of Water Resources - Reservoirs](http://cdec.water.ca.gov/reservoir.html)

Department of Water Resources uses sensors to monitor reservoir levels in California.

|         |            |
| ------------- |-------------|
| Data download      | email |
| Parse before indexing | no  |
| Sourcetypes | cad_reservoir_levels |
| Dashboards | reservoir_levels, reservoir_and_groundwater_levels |
| Lookups | reservoir_location, reservoirs_metadata |
| Datamodel | .  |
| New Data Updated | Every 3-6 months (or whenever they oblige to process the request)  |
| Current Data Available | Jan 2010 - Jun 2015 |
| Data Granularity | month |
| Contacts | welchr@water.ca.gov |

_____

####[USGS Water use data](http://waterdata.usgs.gov/nwis/wu)
USGS performs a survey every five years to aggregate water use for Residential,Industrial and Agricultural purposes.

|         |            |
| ------------- |-------------|
| Data download      | [web](http://waterdata.usgs.gov/nwis/wu) |
| Parse before indexing | no |
| Sourcetypes | cad_usgs_water_use |
| Dashboards | water_use_history |
| Lookups | water_use_FIPS_codes |
| Datamodel | cad_usgs_water_use |
| New Data Updated | 5 years  |
| Current Data Available | 2005 - 2015 |
| Data Granularity | year |
| Contacts | . |

_____

####[CA Census Demograhic data](http://www.census.gov/programs-surveys/acs/data.html)

Data from the Californian Census.

|         |            |
| ------------- |-------------|
| Data download | [web](http://www.census.gov/programs-surveys/acs/data.html) |
| Parse before indexing | yes - [(instructions)](datafiles/data_formatting_process.docx)  |
| Sourcetypes | cad_demographic |
| Dashboards | reservoir_and_groundwater_levels |
| Lookups | . |
| Datamodel | cad_demographic |
| New Data Updated | . |
| Current Data Available | 2011 |
| Data Granularity | year |
| Contacts | . |

_____

####[Weather data](www.wunderground.com/weather/api)

Live and historic data from the Wunderground API.

|         |            |
| ------------- |-------------|
| Data download | [api](www.wunderground.com/weather/api) |
| Parse before indexing | no  |
| Sourcetypes | . |
| Dashboards | . |
| Lookups | . |
| Datamodel | . |
| New Data Updated | . |
| Current Data Available | . |
| Data Granularity | . |
| Contacts | . |

####NOAA Precipitation Data

Shortterm: http://www.ncdc.noaa.gov/cdo-web/search?datasetid=PRECIP_HLY
Longterm: http://www.ncdc.noaa.gov/cag/time-series/us/4/0/pcp/ytd/12/1900-2015?base_prd=true&firstbaseyear=1901&lastbaseyear=2015

## Issues

To raise or view issues with this app, please use the Github repository here.

## Contacts

* Aamir Goriawala (former intern who built most of this project)
* [David Greenwood](dgreenwood@splunk.com)
* [Corey Marshall](cmarshall@splunk.com)

## FAQ

**So what is the difference between ”water use” and “water consumption”?**

* “Water use” describes the total amount of water withdrawn from its source to be used. Measures of water usage help evaluate the level of demand from industrial, agricultural, and domestic users. For example, a manufacturing plant might require 10,000 gallons of freshwater a day for cooling, running, or cleaning its equipment. Even if the plant returns 95 percent of that water to the watershed, the plant needs all 10,000 gallons to operate.

* “Water consumption” is the portion of water use that is not returned to the original water source after being withdrawn. Consumption occurs when water is lost into the atmosphere through evaporation or incorporated into a product or plant (such as a corn stalk) and is no longer available for reuse. Water consumption is particularly relevant when analyzing water scarcity and the impact of human activities on water availability. For example, irrigated agriculture accounts for 70 percent of water use worldwide and almost 50 percent of that is lost, either evaporated into the atmosphere or transpired through plant leaves.
