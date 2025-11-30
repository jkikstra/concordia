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

Done:
- [x] Jarmo explains the current state of the full workflow to Marco and Annika, shows where resources are, and what is still planned
- [x] Marco and Annika attempt to run the main workflow (at least 2 high priority scenarios)
- [x] Jarmo checks Annika's VOC PR (check why some cells are not 0 or 1)
- [x] Jarmo creates ERW proxy
- [x] Jarmo updates variabledefinitions to run multiple-CDR 
    * for Other CDR:
        * tentatively decided to use DAC_CDR as the proxy for Other CDR, which is assumed to mainly be leakages of CDR; for MESSAGE.
        * however, AIM mainly has soil carbon and biochar here, which would be in different locations.
        * other models would still need to be checked.

In progress:
- [~] Jarmo cleans and updates post-processing scripts (including metadata; CO2 sectors; including variable names SO2/Sulfur, NMVOC/VOC; including filenames/scenarionames; including timesteps in netcdf)
    - [x] write a script that checks tha VOC speciation results add up to the total
    - [x] SO2/Sulfur (re)naming in the main workflow 
    - [x] NMVOC renaming (for openburning only) in the main workflow 
        - [ ] check if unit also needs to change with (i would think no)
    - [ ] main workflow files written out as correct as possible; metadata
    - [ ] add global totals in metadata (dressup), e.g.,:
        - [ ] source_id (institute and scenario name need to be passed in correctly)
        - [ ] title (scenario name need to be passed in correctly)
    - [ ] other suggestions from GISS
    - [ ] filename structure in main workflow and streamlined in all files
- [ ] Jarmo ensures that order of post-processing fixes is correct (e.g. CEDS country-border correction also for VOC speciated, after/before 2022?)

Tasks Jarmo:
- [ ] Jarmo works on a (semi-automatic) plotting suites
- [~/x] Jarmo updates multiple-CDR handling for reaggregation
    - [x] include ERW, etc. and
    - [ ] work on biochar and soil carbon sequestration + potentially other land-use (TBD; potentially push to 0-5-0)
- [ ] Annika/Jarmo: zeroes/NAs streamlining between historical-CEDS and our gridded-scenario data for sectors with no/zero emissions.
- [ ] Jarmo: check that fill_value=0 of the ptolemy_patch is working OK 

Tasks Annika:
- [x] VOC biomass
- [ ] update 2022 handling; if no time at all, perhaps consider deleting it altogether (?).
    - [~] 2022 CEDS: 
        * Option A (preferred): remove what we have created, replace with CEDS ESGF file values
        * Option B: remove year altogether in CEDS + update metadata and description in line with that
        * OPtion C:   remove year altogether in CEDS as well as removing it in openburning (to try to avoid having different years in our dataset, letting everyone just interpolate between 2021 and 2023 which shouldn't be too bad)
    - [~] 2022 openburning: keep as is, just check that it looks OK

Tasks Marco:
- [ ] understand how to run everything, perform test runs
- [ ] perform run for H and VL, run all plotting, and check that the data is correct (share the plots with Jarmo too)
- [ ] consider ways to run (on unicc?), to deal with storage space issues (if necessary)
- [ ] input4mips (metdadata) validation script; see README.md in CMIP7 notebooks, section "Validation: data format checking for input4MIPs" -- ensure that the netCDF metadata is correct for upload to ESGF
- [ ] ensure information for biochar and soil carbon sequestration CDR are pre-processed for all IAMs (focus first on the high-priority scenarios). See main notes here: https://docs.google.com/document/d/1H9sKOkTLC1oDxEWUNurXqilkEoz3obH5J5rk5PCTXvk/edit?tab=t.ozl8f8vi3kgh (with email by Jarmo on November 26, 2025 04:09). 


DELETE: Futher (older) notes: we need to include these emissions (removals) in the emissions reporting. The reason is that, even though it is land-based, it is not included in any land-use/land-use change variable product, and thus ESMs won't be able to tell anything about these emissions from the land-use product provided by Louises, thus we must include it on the emissions side. So far we haven't included this information. It should first be added in gcages, then in harmonization (harmonize as zero, just like other CDR methods that are not covered in CEDS), then be included in the gridding workflow (similar to other CDR like ERW) and then be aggregated into other CDR. Gridding proxy files still need to be determined. Could use AGLAND (agricultural land) from the RESCUE file as a (good enough?) starting point. 
    - Some information per model (collected by Jarmo so far):
        - AIM: biochar and soil carbon sequestration are reported under 'Emissions|CO2|Other Capture and Removal'
            - Emissions: not under CO2|AFOLU, but under 'Emissions|CO2|Other Capture and Removal'
            - Carbon Removal: don't use, as it is under the 'Emissions' tree where we are already gridding it
        - COFFEE: Luiz Bernardo says that Gerd Angelkorte will make sure that this will be separated in the upcoming submission
            - Emissions: ...
            - Carbon Removal:
                - ...
                - ...
        - GCAM: ...tbd - ask Steve Smith / Xin Zhao / Christoph Bertram / Mel George
            - Emissions: ...
            - Carbon Removal:
                - ...
                - ...
        - IMAGE: does not have biochar or soil carbon sequestration
            - Emissions: N/A
            - Carbon Removal: N/A
        - MESSAGE: ...tbd - ask Yazhen Wu/Oliver Fricko/Yoga Pratama
            - Emissions: ...
            - Carbon Removal:
                - ...
                - ...
        - REMIND: Laurin says it is probably reported under CO2|AFOLU, Florian Humpenöder is yet to confirm
            - Emissions: under AFOLU
            - Carbon Removal: yes
                - no 'Carbon Removal|Land Use|Biochar'
                - Carbon Removal|Land Use|Soil Carbon Management
        - WITCH: 
            - Emissions: under AFOLU
            - Carbon Removal: yes 
                - Carbon Removal|Land Use|Biochar (NOTE: reporting misses a few model years of data) 
                - Carbon Removal|Land Use|Soil Carbon Management
    - related notes:
        - not yet sure what to do with Agroforestry (check with Louise/Detlef/Ben/others: how would this be best included in ESMs)
        




Other (minor) things to be decided:
- [x] decision on Ukraine (esp. transportation): war emissions from transportation are only <5% of what they were pre-war; can we do a believable fix in harmonization? --> DECISION: don't try to patch this for v0-4-0
- [~] double-check soil carbon sequestration and biochar; which models do it and where do they put it. If under AFOLU (AIM under Other Capture and Removal) then we may need to adjust gridding. 
- [ ] Fix small countries issue: https://github.com/iiasa/emissions_harmonization_historical/issues/30#issuecomment-3574985850
    - [ ] double-check history treatment of sdn_ssd, isr_pse, srb_ksv

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
