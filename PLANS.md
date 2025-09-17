# TODO before merge; CMIP7 output checks

- [x] merge commit "Commit 7430abc"; just merge the branch before I start changing stuff


# Notebooks folder

## Documentation
- [x] Write down workflow for a new version of CMIP7-ScenarioMIP data.
  - [x] write down for esgf-based workflow as notes
  - [x] put it in a README.md

## Naming logic (for the notebooks folder) for cmip7:
**Note: keeping rescue filenames untouched, until we speak with Matt?** 

__File naming structure__
- {project-name} / "{type-of-file}_{description-of-purpose-or-action}_{version}"
  - {project-name} is not necessary when the script or data is generic and can be used (unchanged) for more than 1 project. If the file is not meant to be used (unchanged) for more than that specific project, put it in the project folder. Future code development can always later choose to move it back to the root notebooks folder when made more generic. 
  - {version} should be avoided, but may be added if there is a good reason to keep multiple versions for the same project, for instance to enable running the same workflow with multiple configurations 
  - the file should not end with "_"

__Main configuration files__
- "config_*": main configuration file, can serve all types of scripts

__Main python files__
- "prep_*": before running "workflow_*", includes:
  - "prep_proxy_*"
- "workflow_*": from input (IAM {harmonized or raw} emissions) to downscaled and gridded dataproducts 
- "workflow-postprocess_*": does additional processing on the spatial grid files after "workflow_*", for instance adding a year at the start 
- "check_*": checks on all data created in "prep_*" and "prep_*". checks can be both numerical and visual (plots) 

__Helper files__
- "investigate_*": notebooks that look into input files. These files do not produce anything needed for "workflow_*", nor do they analyse the outputs of "workflow_*"



## Moving/current files:

Legend:
[ ] not yet done, still to do
[~] moved but code and names still need to be adjusted
[x] done 

### To delete [DONE]
- gridding_data
  - generate_ceds_proxy_netcdfs_cmip7.py
- config_cmip7_v0_testing.yaml
- workflow-cmip7.py / ipynb
- workflow-cmip7_explore_grids_ceds /ipynb
- workflow-cmip7_investigate-CEDS-raw-proxy-file /ipynb

### To keep
cmip7
- [x] prep_countrymask-from-ceds-pnnlRd.py (previously: gridding_data / create_countrymask_indexraster.py)
- [x] prep_proxyinput-download-anthro-pnnl.py (previously: workflow-cmip7-download-CEDS-raw-proxy-files.py)
- [x] prep_proxyfuture-anthro-from-ceds-cmip7-pnnlRd.py (previously: gridding_data / generate_ceds_based_future_proxy_netcdfs-cmip7.py) # NOTE: remove open_burning, proxyraster
- [x] prep_proxyfuture-anthro-from-ceds-cmip7-esgf.py (previously: gridding_data / generate_proxies_from_CMIP7_CEDS.py)
- [x] prep_proxyfuture-openburning-from-dres-cmip7-esgf.py (previously: gridding_data / generate_ceds_based_future_proxy_netcdfs-cmip7.py) # NOTE: only keep as an empty file because open_burning is not currently functional for cmip7 yet
- [x] prep_proxyfuture-cdr-from-rescue.py (previously: investigate_cdr_cmip7.py)


----
- [x] workflow-cmip7-fast-track.py (previously: workflow-cmip7-already_region_harmonized.py)
----
- [x] check_gridded-scenarios-country-level-reaggregation-to-cedsCountry.py (previously: check_country_level_aggregation.py)
- [x] check_gridded-scenarios-cmip7-vs-cmip6.py (previously: check_gridded_CMIP7_vs_CMIP6.py)
- [x] check_gridded-scenarios-Aircraft-latitude.py (previously: check_gridding_Aircraft-latitude.py)
- [x] check_gridded-scenarios-global-sectoral-aggregation-compared-to-input.py (previously: check_gridding_scenarios_outputs.py) # creates stuff like sectoral_reaggregated_gridded_* plots
- [x] check_gridded-scenarios-compare-to-ceds-esgf.py (previously: compare_to_CEDS_ESGF.py)
- [x] check_plot-animated-grids.py (previously: plot_animated_grids.py)
----
- [x] config_cmip7_v0_testing_ukesm_remind.yaml
- [x] config_cmip7_v0_2.yaml
- [x] README.md # describe the CMIP7 workflow 
----
- [x] investigate_bb4cmip7.py (previously: investigate_biomassburning_cmip7.py)
- [x] investigate_CEDS-pnnlRd.py (previously: investigate_CEDS_Rd_files.py)
- [x] investigate_country-level-aggregation-cedsESGF-vs-cedsCountry.py (previously: check_country_level_aggregation-CEDS-ESGF.py)
- [x] investigate_seasonality-files.py (previously: gridding_data / produce_seasonality_ncfiles.py)


- not-in-use # can include (a) [recently] deprecated scripts (b) scripts that may be [re]used but just need a fix to work again
  - [x] build-harmonization-report-cmip7.py
  - [x] investigate_maps_multipanel.py # new; started as plotting CDR maps 
  - [x] check_gridding_proxy.py # new; some quick plots of proxy files

rescue
- historical_data 
  - [x] extract_per_country_gfed_from_hdf5.ipynb
- harmonization_postprocessing
  - [x] process_future_hfcs.py
- gridding_data
  - [x] renewable_potential_from_gasp_gwa_gsa.py
  - [x] generate_ceds_proxy_netcdfs.py 
      [ ]  # NOTE: reverse settings commit b7b594d https://github.com/jkikstra/concordia/commit/b7b594d661e0d962b3def93a4671705a37c7d8c9 e.g. with an if-statement for project
  - [x] generate_non_ceds_proxy_netcdfs.py 
      [ ] # NOTE: change commit b7b594d https://github.com/jkikstra/concordia/commit/b7b594d661e0d962b3def93a4671705a37c7d8c9 e.g. with an if-statement for project 
  - [x] renewable_potential_from_gasp_gwa_gsa.py
- [x] build-harmonization-report-rescue.py
- [x] config-rescue.yaml
- [x] local-config-rescue.yaml 
- [x] prepare-variabledefinitions.py
- [x] workflow-rescue.py

generic
- build-harmonization-report.py
- example_config.yaml
- README.rst
- workflow.py
- workflow_cmip6.py # keep it here as example


## Checks and plots: code changes

- [ ] Structure the checks and their outputs a bit more
- [ ] Move plotting functions that are otherwise duplicated to a new plotting utils file
- [ ] Create a 'checks_and_plots' run file



