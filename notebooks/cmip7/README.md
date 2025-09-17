# CMIP7 workflow (fast-track)

*Last update: 2025-09-13*

This page describes how to produce a new version of emissions grids for the CMIP7 fast-track.

All mentioned files are located in this folder, unless mentioned otherwise.

## Starting point
It is assumed that you have a pre-harmonised scenario.

## Current Main workflow

### Configuration

**TODO**: for each version, need to specify data paths for input and output:
* `config_cmip7_{version}.yaml` 
  * `variabledefs_path`: this CSV file is important. It specifies all species-sector combinations that are produced, and which method is used (global/country, which proxy file and proxy variable) and what output variable/file the data will be located in (column: "output_var").
  * `history_path`: CEDS + GFED (country-level), which is produced in the repo `iiasa/emissions_harmonization_historical`, speficically [COUNTRY_LEVEL_HISTORY](https://github.com/iiasa/emissions_harmonization_historical/blob/a59ad18ac3e2f6a15c2bd354fabb85c40cd60b91/notebooks/5011_create-history-for-gridding.py#L96). **N.B.** currently it is important that this is exactly the same as the data used in the harmonization process in [5094_harmonisation.py](https://github.com/iiasa/emissions_harmonization_historical/blob/main/notebooks/5094_harmonisation.py)
  * `scenario_path`: where are (harmonised) IAM scenarios located
  * `regionmappings_path` & `regionmappings`: for each IAM, we need to know which countries are covered by which region. **N.B.** currently model-names (including their version numbers) are hard-coded and thus it needs to be checked that this is correct. 
  * `gridding_path`: hosts input files for gridding (many which need to be downloaded before starting the further processing)
  * `proxy_path`: hosts spatial proxies for gridding, which are produced in the "pre-processing" step described below 

**TODO**: Make sure that your data folder has the following:
* ...
* ...

### Pre-processing

Run first:
1. `prep_countrymask-from-ceds.py`: creates the countrymask (from CEDS .Rd mask files)

After that, prepare the proxy files for future years:
1. `prep_proxyfuture-anthro-from-ceds-cmip7-esgf.py`: prepares proxies for (land-based) anthropogenic emissions, shipping, and aircraft
1. `prep_proxyfuture-openburning-from-dres-cmip7-esgf.py`: prepares proxies for openburning emissions
1. `prep_proxyfuture-cdr-from-rescue.py`: prepares proxies for CDR

**TODO:**
- [ ] Make `prep_proxyfuture-openburning-supplemental-VOCspeciation-from-dres-cmip7-esgf.py`
- [ ] Make `prep_proxyfuture-anthro-supplemental-VOCspeciation-from-ceds-cmip7-esgf.py`


### Workflow

1. `workflow-cmip7-fast-track.py`

**TODO:**
- [ ] update `workflow-cmip7-fast-track.py` to start the scenario data in 2022 (need to extend the scenario data with one year of history data)

### Post-processing

**TODO:**
- [ ] Make `workflow-postprocess_harmonize-output-grids-to-cmip7-ceds-esgf.py...`: a script that "glues together" our scenarios and the CEDS ESGF files, for the harmonization year (2023).
- [ ] Make `workflow-postprocess_add-missing-years-cmip7-ceds-esgf.py`: makes the year 2022 for ceds; just copy the 2022 files from CEDS, and add them to our scenario files in the same format as our scenario files.
- [ ] Make `workflow-postprocess_add-missing-years-cmip7-bb4cmip7-esgf.py`: makes the year 2022 for bb4cmip7; **N.B. consider whether this also needs a "force-fix": make also 2021 and see whether we are wrong there or not**

### Checking and plotting

Main checking scripts:
* `check_gridded-scenarios-global-sectoral-aggregation-compared-to-input.py`: checks global totals of output grids against input harmonized scenario
* `check_gridded-scenarios-compare-to-ceds-esgf.py`: compares all sectors of outputs to CEDS grids in harmonization year (2023), and also can make many timeseries plots for specific points and areas (though please note that this can take very long)
* `check_gridded-scenarios-country-level-reaggregation-to-cedsCountry.py`: checks whether our output grids align with the input historical national emissions data of CEDS

Other checking scripts that do not always need to be run:
* `check_plot-animated-grids.py`: makes animated grids (is very useful, especially for presentations, but also takes very long)
* `check_gridded-scenarios-cmip7-vs-cmip6.py`: checks timeseries of our output grids to cmip6 scenario output grids 
* `check_gridded-scenarios-Aircraft-latitude.py`: checks latitudinal profile of aircraft emissions

## Other

### Potential other choices in workflow
Instead of creating the CEDS-anthro proxies from the ESGF-located .nc files, it is also possible to construct proxies from intermediary .Rd files provided by the PNNL/JGCRI/CGS team on [their server](https://rcdtn1.pnl.gov/data/CEDS/Jarmo_files/). 

In that case, you would want to skip the `` notebook, and instead run:
1. `prep_proxyinput-download-anthro-pnnl.py`: download proxy data from PNNL's server
2. `prep_proxyfuture-anthro-from-ceds-cmip7-pnnlRd.py`: make the proxies based on files from PNNL's server (**N.B.** this does not currently work for aircraft. It only works for anthro and shipping, and it does not create particularly good grids for anthro due to mistakes in the input data.)



### Occasional-use scripts with functionality that is sometimes useful

* `investigate_seasonality-files.py`: load and plot CEDS seasonality .Rd files from the PNNL server
* `investigate_bb4cmip7.py`: load and plot BB4CMIP7 files
* `investigate_CEDS-pnnlRd-files.py`: load and plot CEDS .Rd files from the PNNL server
* `investigate_country-level-aggregation-cedsESGF-vs-cedsCountry.py`: compare CEDS grids by aggregating them to the country level, to the already national CEDS data which is input for harmonisation
