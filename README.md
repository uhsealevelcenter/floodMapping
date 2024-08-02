# Mapping Flood Frequency

Major goal: put Phil's flood frequency analysis into space. 
(Launch it to the mooooon!)

We're focusing on Kennedy Space Center for now.

Idea: We have the number of days flooded per year, and we want to map where these thresholds are reached.

Using: [NOAA SLR DEMs](https://coast.noaa.gov/slrdata/), [VDatum](https://vdatum.noaa.gov/welcome.html) (for a tidal surface if applicable), and Phil's [tide gauge analysis](https://sealevel.nasa.gov/flooding-analysis-tool/projected-flooding?station-id=8721604&scenario=int-low&threshold=061).

At Kennedy Space Center, that means data from the Trident Pier tide gauge and the water level gauge from Haulover canal.
The tidal surface is developed following NOAA's SLR viewer documentation.

Looking at the intermediate-low scenario, we're basically taking this plot (for both stations):

![Screenshot](vizForMarkdown/floodanalysistool.png)

to these plots:

<div style="display: flex; justify-content: space-around;">
    <img src="vizForMarkdown/int_low_threshold_to_days_2020.png" alt="Flood Mapping 2020" style="width: 50%;">
    <img src="vizForMarkdown/int_low_threshold_to_days_2070.png" alt="Flood Mapping 2070" style="width: 50%;">
</div>


and turning it into this:

![Animation](vizForMarkdown/htf_int_low_animation_MHHW.gif)

