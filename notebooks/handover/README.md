# Burn severity mapping -- ANU-WALD
This repository is a collection of scripts used to produce the automate burn severity mapping using the Digital Earth Australia.

In this repository, there are tthree ~.py providing modules for burn mapping and validation use. Each module was documented. Use help(module_name) for more details.

The details of each script can be found in the following section.

## Scripts
BurnCube.py defines the class for multiple functions used in the burn mapping, e.g. geomedian,distance,severity

stats.py defines the statistical tools used in several functions in the BurnCube class

validationtoolbox.py includes several modules used for the validation of detected burned area, such as the calculation of roc analysis

Burn_Mapping_BurnCube_Example1_(DEA).ipynb  provides an example of using the burnmappingtoolbox for burn mapping and details of the methods and the workflow of the burn mapping.

Validation_example1.ipynb provides an example of using validationtoolbox for validating burn mapping with fire perimeters polygons

mycolormap.txt is a predefined colormap used for the severity mapping 

## Validation data
The validation dataset is currently located under /g/data/xc0/project/Burn_mapping/02_Fire_Perimeters_Polygons/