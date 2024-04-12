# MAKE CSV for mapping KSC Locations
# Hard coded because that's how I roll
KSC_launch39A = [538682, 3164664]
KSC_visitors_center = [531143.87, 3155058.75]
KSC_tidegauge = [539834, 3143243]

# make a csv with name, easting, northing
import csv
with open('KSC_locations.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Easting', 'Northing'])
    writer.writerow(['Launch Pad 39A', KSC_launch39A[0], KSC_launch39A[1]])
    writer.writerow(["Visitor's Center", KSC_visitors_center[0], KSC_visitors_center[1]])
    writer.writerow(['Trident Pier Tide Gauge', KSC_tidegauge[0], KSC_tidegauge[1]])
