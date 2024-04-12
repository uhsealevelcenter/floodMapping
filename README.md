# Mapping Flood Frequency

Major goal: put Phil's flood frequency analysis into space. 
(Launch it to the mooooon!)

We're focusing on Kennedy Space Center for now.

Idea: We have the number of days flooded per year, and we want to map where these thresholds are reached.

Using: NOAA SLR DEMs, vDATUM (for a tidal surface), and Phil's tide gauge analysis.

At Kennedy Space Center, that means data from the Trident Pier tide gauge.

Looking at the intermediate-low scenario, we're basically taking this plot:

![Screenshot](viz/floodanalysistool.png)

to this plot:

![Flood Mapping](viz/int_low_threshold_to_days.png)

and turning it into this plot:

![QGIS Layer](viz/low_2100.png)

