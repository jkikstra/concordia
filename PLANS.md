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
- Jarmo explains the current state of the full workflow to Marco and Annika, shows where resources are, and what is still planned

29.10.2025: 
- Marco and Annika attempt to run the main workflow (at least 2 high priority scenarios)
- Jarmo cleans and updates post-processing scripts (including metadata; including variable names SO2/Sulfur; including filenames/scenarionames; including timesteps in netcdf)
- Jarmo checks Annika's VOC PR (check why some cells are not 0 or 1)
- Jarmo creates ERW proxy

30.10.2025: 
- Meeting (online) to do Q&A and distribute tasks

31.10.2025:
- Jarmo works on a (semi-automatic) plotting suite

Tasks to be distributed:
- consider ways to run (on unicc?), dealing with storage space issues
- update 2022 handling
- multiple-CDR handling
- input4mips (metdadata) validation script

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
