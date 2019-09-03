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


## Output description
Outputs include two classes for burned area, namely high-severity and moderate-severity burns. The severity of a burned area is calculated by the temporal integral of cosine distance for the period of time that it remains a statistically significant anomaly compared to the other spectra in the time series. The moderate-severity burns is deteced using the region-growing method. This include further areas that do not qualify as outliers but do show a substantial decrease in NBR and are adjoining pixels detected as severe burns.

The outputs of severity and burnscar mapping are stored in one dataset and saved in netcdf format. The following variables are included in the output file:

- StartDate (float64): detected start-date of severely and moderately burned area (nan for unburnt area)
- Duration (int16): duration of land cover change due to fire for severely burned area (0 for unburnt area)
- Severity (float32): severity of land cover change due to fire for severely burned area  (0 for unburnt area)
- Severe (int16): binary mask for severely burned area (0 for unburnt area)
- Moderate (int16): binary mask for severely and moderately burned area (0 for unburnt area)
- Corroborate (int16): binary mask for corroborating evidence from hotspots data with 4km buffer (0 for unburnt area)
- Cleaned (int16): month of StartDate for severely and moderately burned area that's spatially connected and consistent with hotspots data (0 for unburnt area)

## Validation data
The validation dataset is currently located under /g/data/xc0/project/Burn_mapping/02_Fire_Perimeters_Polygons/

## Run it on raijin
The 'burnmapping_test.py' provides the example of running the BurnCube for a 25km tile with the centre lat,lon coordinates for a list of selected sites in sites.txt. 


The 'jobs.pbs' provides a simple example of scheduling a job for one tile in raijin calling the 'burn_mapping_tiles.py'.
    To submit a single job for one tile, sepecify the tileid, mapping year, method, directories as follow:
    e.g. qsub -v ti=450,year=2017,method='NBR',dir='/test/',subdir='/test/subtiles/' jobs.pbs


The 'scheduler.py' set up the runs for multiple 100km tiles by submitting multiple 'jobs.pbs' with the given tile indices.Please note that it will check the existence of subtiles and merged tiles before submitting the jobs to raijin. This script will submit one job per tile to raijin. 

    To process muliple tiles:
    1. specify the mapping year, mapping method, change the directories for the 25km subtiles and merged 100km tiles in 'scheduler.py'. 
    2. pass the index number to scheduler and submit the jobs to raijin: 
    e.g. python scheduler.py 400,500


