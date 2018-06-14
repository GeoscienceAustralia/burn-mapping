# Burn severity mapping -- ANU-WALD
This repository is a collection of scripts used to produce the automate burn severity mapping using the Digital Earth Australia.

In this repository, there are two ~toolbox.py providing modules for change detection mapping and validation use. Each module was documented. Use help(module_name) for more details.

The details of each script can be found in the following section.

## Scripts
loaddea.py includes the modules for loading DEA landsat data and masked out the cloudy pixels

burnmappingtoolbox.py  includes several modules used for burn mapping, such as the calculation of geometric median, cosine distance

validationtoolbox.py includes several modules used for the validation of detected burned area, such as the calculation of roc analysis

disttogeomedian.py and changedetection.py can be used directly for a 2D region for the calculation of geometric medians and severity. These two scripts already include the use of modules in the burnmappingtoolbox.

severitymapping.py can be directly used for severity mapping by given the geographic extent and other required information, and can be easily adapted for the continental process with paralle computation. 

Burn_Mapping_Detailed_Example1_(DEA).ipynb  provides an example of using the burnmappingtoolbox for burn mapping and details of the methods

Burn_Mapping_Simple_Case_Example1.ipynb provides an example of the simple workflow of using disttogeomedian and changedetection for mapping severity. 

Validation_example1.ipynb provides an example of using validationtoolbox for validating burn mapping with fire perimeters polygons

mycolormap.txt is a predefined colormap used for the severity mapping 

## Validation data
The validation dataset is currently located under /g/data/xc0/project/Burn_mapping/02_Fire_Perimeters_Polygons/