# Burn severity mapping -- ANU-WALD
Landsat-Based Burn Extent and Severity Mapping

This collaborative project will use the Digital Earth Australia (DEA) data
infrastructure to develop an automated algorithm for automated mapping of burnt area extent from
DEA Landsat that is suitable for Australia-wide deployment, and can be used to determine burn
severity and fire frequency.

Some states have automated or semi-automated methods for rangeland burn extent
mapping. However, in forests, burnt area extent and burn severity mapping is currently usually done
ad hoc after major events.

Data used in mapping normally include satellite imagery enhanced with
on-ground mapping and insights, using mapping techniques that are fine-tuned to suit the
characteristics of the event and data. This approach produces appropriate results for the event at
hand, but does not produce a longer burn history, which is needed to understand current and future
fire risk.

There is a clear need for automated techniques for mapping burnt area extent and severity
and fire risk that can be applied anywhere in Australia, including in woody vegetation systems.
Continuous mapping of burnt area will also help to inform and attribute land cover change mapping
carried out by state and Commonwealth agencies (e.g. NCAS).

Project Objective: To develop automated algorithms that use data contained in the Digital Earth
Australia data infrastructure to map burnt area extent in a manner that is suitable for Australia-
wide operationalisation, with a focus on woody vegetation. The method and data will be
validated against events for which independent spatial data are available. Techniques will be
developed to calculate fire frequency from the burnt area extent mapping.

This repository is a collection of scripts used to produce the automate burn severity mapping using the Digital Earth Australia. In this repository, there are tthree ~.py providing modules for burn mapping and validation use. Each module was documented. Use help(module_name) for more details.

The details of each script can be found in the following section.

## Scripts
BurnCube.py defines the class for multiple functions used in the burn mapping, e.g. geomedian,distance,severity

stats.py defines the statistical tools used in several functions in the BurnCube class

validationtoolbox.py includes several modules used for the validation of detected burned area, such as the calculation of roc analysis

Burn_Mapping_BurnCube_Example1_(DEA).ipynb  provides an example of using the burnmappingtoolbox for burn mapping and details of the methods and the workflow of the burn mapping.

Validation_example1.ipynb provides an example of using validationtoolbox for validating burn mapping with fire perimeters polygons

mycolormap.txt is a predefined colormap used for the severity mapping 

## Output description
Outputs include two classes for burned area, namely high-severity and moderate-severity burns. The high severity burns is calculated by the temporal integral of cosine distance for the period of time that it remains a statistically significant anomaly compared to the other spectra in the time series. The moderate-severity burns is deteced using the region-growing method. This include further areas that do not qualify as outliers but do show a substantial decrease in NBR and are adjoining pixels detected as burns.

The outputs of severity and burnscar mapping are stored in one dataset and saved in netcdf format. The following variables are included in the output file:
    StartDate (float64): detected start-date of severe and moderate burned area (filled with nan for unburnt area)
    Duration (int16): duration of land cover change due to the bushfire (filled with nan for unburnt area)
    Severity (float64): severity of land cover change due to the bushfire (0 for unburnt area)
    Severe (int8): binary mask for severe burnt area (0 for unburnt area)   
    Moderate (int8): binary mask for moderate and severe burnt area (0 for unburnt area)  
    Corroborate (int8): binary mask for corroborating evidence from hotspots data with 4km buffer (0 for unburnt area) 


## Validation data
The validation dataset is currently located under /g/data/xc0/project/Burn_mapping/02_Fire_Perimeters_Polygons/

## Run it on raijin
The 'burnmapping_test.py' provides the example of running the BurnCube for a 25km tile with the centre lat,lon coordinates. The compuation tile is recorded for each step.

The 'jobs.pbs' provides a simple example of scheduling the job for one tile in raijin

The 'testsites_run.sh' provides a simple example of submitting muliple jobs to raijin that mapping the burnscar for the sites in sites.txt