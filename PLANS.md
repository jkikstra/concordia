# Version: 0-4-0

A short 1-stop overview of what needs to be done for version 0-4-0. 

## Code overview changes to be implemented:

GitHub Pull Request: https://github.com/jkikstra/concordia/pull/15

## Data location

IIASA Sharepoint "concordia_cmip7_v0-4-0": https://iiasahub.sharepoint.com/:f:/r/sites/eceprog/Shared%20Documents/Projects/CMIP7/IAM%20Data%20Processing/concordia_cmip7_v0-4-0?csf=1&web=1&e=TUvLMQ

## README

notebooks\cmip7\README.md

## Running

Main driver to integrate all scripts (work in progress):

scripts\cmip7\driver_workflow_cmip7-fast-track.py

### Proxies

- [ ] aim to rerun all proxies to ensure we can (re)create them

### Workflow

- [ ] aim to rerun all proxies to ensure we can (re)create them

### Checks

- [ ] a plotting suite as complete as possible (to convince ourselves that we don't have any errors)

## Timeline

28.10.2025: 
- [x] Jarmo explains the current state of the full workflow to Marco and Annika, shows where resources are, and what is still planned

29.10.2025: 
- [x] Marco and Annika attempt to run the main workflow (at least 2 high priority scenarios)
- [~] Jarmo cleans and updates post-processing scripts (including metadata; CO2 sectors; including variable names SO2/Sulfur; including filenames/scenarionames; including timesteps in netcdf)
- [x] Jarmo checks Annika's VOC PR (check why some cells are not 0 or 1)
- [x] Jarmo creates ERW proxy

31.10.2025:
- [ ] Jarmo works on a (semi-automatic) plotting suite

Tasks Annika:
- [ ] VOC biomass
- [ ] update 2022 handling; if no time at all, perhaps consider deleting it altogether (?).
    - [ ] 2022 CEDS: 
        * Option A (preferred): remove what we have created, replace with CEDS ESGF file values
        * Option B: remove year altogether in CEDS + update metadata and description in line with that
        * OPtion C:   remove year altogether in CEDS as well as removing it in openburning (to try to avoid having different years in our dataset, letting everyone just interpolate between 2021 and 2023 which shouldn't be too bad)
    - [~] 2022 openburning: keep as is, just check that it looks OK

Tasks Marco:
- [ ] understand how to run everything, perform test runs
- [ ] perform run for H and VL, run all plotting, and check that the data is correct (share the plots with Jarmo too)
- [ ] consider ways to run (on unicc?), to deal with storage space issues (if necessary)
- [ ] input4mips (metdadata) validation script; see README.md in CMIP7 notebooks, section "Validation: data format checking for input4MIPs" -- ensure that the netCDF metadata is correct for upload to ESGF

Tasks to be distributed:


Tasks for Jarmo:
- [~: is in main workflow, not yet in the sector reaggregation] multiple-CDR handling
    * tentatively decided to use DAC_CDR as the proxy for Other CDR, which is assumed to mainly be leakages of CDR. 
- [x] decision on Ukraine (esp. transportation): war emissions from transportation are only <5% of what they were pre-war; can we do a believable fix in harmonization? --> DECISION: don't try to patch this for v0-4-0
- [~] double-check soil carbon sequestration and biochar; which models do it and where do they put it. If under AFOLU (AIM under Other Capture and Removal) then we may need to adjust gridding. 

### Update 30.10.2025

#### Marco:
Installing environment with mamba worked fine. Just added jupytext.

main workflow:
Running REMIND main workflow worked fine.

post-processing:

* post-processed spatial harmonization requires having the data locally.
* post-processed reaggregation also worked
* note: need to streamline 'GRDIDDING_VERSION' etc.
* note: 'return_marker_information' -- needs to be streamlined also, take the right config as parameter

Next to-do:
* Plots; `check_gridded-scenarios-global-sectoral-aggregation-compared-to-input.py` & `check_gridded-scenarios-compare-to-ceds-esgf.py` (note: check if sectors are different, or not)
* parameters & driver for post-processing; `workflow-postprocess_anthro-pattern-harmonisation.py` & `workflow-postprocess_anthro-reaggregate-CDR-sectors.py` 

#### Annika:
Ran driver, that worked. 

-----------------------
-----------------------
**Legend of TODOs**:

[ ] not yet done, still to do

[~] moved but code and names still need to be adjusted

[x] done 
