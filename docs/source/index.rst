

Getting Started
===============

We suggest using environments to install this package, and in particular using
[`mamba`](https://mamba.readthedocs.io/en/latest/) for the fastest installation
experience (though `conda` and `pip install` work fine too).

mamba/conda
-----------

.. code-block:: bash
    $ mamba env create -f environment.yml

*OR* pip
--------

.. code-block:: bash
    $ pip install -e .[lint,rescue,test]

You can then check to make sure your install is operating as expected

.. code-block:: bash

    pytest tests

Why aren't there notebook files?
--------------------------------

In order to be able to properly track versions and changes, all notebooks here
are in the form of python files, which can be synced as jupyter notebooks with
the [jupytext](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html)
tool, e.g.,

.. code-block:: bash

    jupytext --sync workflow.py # generates a workflow.ipynb file

If you make a material change to `workflow.ipynb`, you can resync it manually if
needed with

.. code-block:: bash

    jupytext --sync workflow.ipynb # updates the workflow.py file

Anatomy of a Workflow
=====================

The purpose of a workflow is to support the harmonization of IAM emission data
to a compiled historical data set, downscaling of that data, and gridding of
that data. An example of such a workflow is provided in
[`notebooks/workflow`](https://github.com/IAMconsortium/concordia/blob/main/notebooks/workflow.py)

Concepts
--------

TODO: describe harm, downscale, grid and different combinations of regional resolution

See https://docs.google.com/drawings/d/1XsvCyIgC3pjaWSgZLPuYDsrjEVzdldHVhbEtrKN7xEE/edit?usp=sharing

Technical overview of a workflow object
---------------------------------------

TODO: write what objects and functions are part of the WorkflowDriver class.

WorkflowDriver [concordia.workflow.WorkflowDriver]

- model: pd.Dataframe
- hist: pd.Dataframe
- gdp: pd.Dataframe
- regionmapping: concordia.RegionMapping
- indexraster_country: ptolemy.IndexRaster
- indexraster_region: ptolemy.IndexRaster
- variabledefs: concordia.VariableDefinitions
- harm_overrides: pd.Series[str]
- settings: concordia.settings.Settings


Configuration
-------------

All workflows require a configuration file, and an example is provided in
[`notebooks/example_config.yaml`](https://github.com/IAMconsortium/concordia/blob/main/notebooks/example_config.yaml).

Defining Variables
------------------

TODO: describe variabledefs

Data Requirements
-----------------

TODO: describe historical and model data and proxies

Harmonization
-------------

TODO: describe briefly harmonization concepts and point to aneris docs

Downscaling
-----------

TODO: describe briefly downscaling concepts and point to aneris docs (need to
write aneris docs on downscaling methods)

Gridding
--------

Original process:
-> gridding files
    * original: https://github.com/iiasa/emissions_downscaling/blob/79dc01346dd841b0cff82f83df31706d6995c94c/code/parameters/gridding_functions.R#L98
    * now referred to in generate_non_ceds_proxy_netcdfs (line 107)



TODO: Describe dask client, proxy data, etc.

Postprocessing
--------------

TODO: describe f-gas approach

Uploading Data
--------------

TODO: describe FTP updates

Analyzing Results
=================

The most important part of a workflow related to matching the original provided
scenario data is harmonization. As a result, we provide a specific utility -
[`build-harmonization-report`](https://github.com/IAMconsortium/concordia/blob/main/notebooks/build-harmonization-report.py)
which constructs graphs for every processed trajectory. The resulting output can
be generated as a static `html` file which can be served and reviewed.

Adding a New Sector
===================

To add a new sector that can be processed, we need 3 things:

1. a spatial proxy dataset
2. scenario data
3. (optional) historical data

In addition to this, we need to make a few changes in the code, configuration,
and data to accomodate the new sector.

1. Add spatial-proxy generating code (e.g., in
`concordia/notebooks/gridding_data/generate_non_ceds_proxy_netcdfs`)

.. tip:: You can skip this part if you already have a proxy generated that works for your sector (e.g., non-urban land area)

2. Add a line per relevant emissions species in your
   `data/variabledefs-<myproject>.csv` that identifies the downscaling
   resolution you want (column: `griddinglevel`) and set the values of
   `proxy_path` `output_variable`, and `proxy_sector` consistent with what you
   did in **Step 1**

3. Add your sector in the preferred order in `concordia/src/concordia/<my
   project>/utils.py`. The sector name here **must be identical** to the value
   in the `sector` column for the line(s) you added in **Step 2**

4. Rerun your `notebooks/workflow-<myproject>` workflow file to make sure
   everything runs smoothly


Key Downstream Depedencies
==========================

`concordia` has a number of primary downstream depedencies which serve different
purposes.

For working with large-scale gridded data computations, we take advantage the
`xarray` ecosystem, including `dask` and `flox`.

For special operations translating between vector data and grids, we use
[`ptolemy`](https://github.com/gidden/ptolemy), including its `IndexRaster`
implementation.

`concordia` operationalizes three main processes: harmonization of IAM emission
data, downscaling of that data, and gridding of that data. `concordia` tries to
stay lean in terms of just connecting the pieces together, but all
implementation for these different processes lives in
[`aneris`](https://github.com/iiasa/aneris).


Applications
==========================


RESCUE workflow Documentation
-----------------------------

Data location: (IIASA SharePoint)/RESCUE - WP 1/data_2024_09_16 

Script: notebooks/workflow-rescue.py

Configuration file: local-config-rescue.yaml (based on config-rescue.yaml)

Example results: concordia/results/TEST-2025-04-10/harmonization-TEST-2025-04-10.csv

WorkflowDriver [concordia.workflow.WorkflowDriver] inputs: 

- ``model`` (pd.Dataframe)
    - line of code: 
        - ``pd.read_csv(settings.scenario_path / "REMIND-MAgPIE-CEDS-RESCUE-Tier1-2024-10-11.csv",index_col=list(range(5))sep=";") ...``
        - input file: settings.scenario_path / "REMIND-MAgPIE-CEDS-RESCUE-Tier1-2024-10-11.csv"
    
    - MultiIndex: ('model', 'scenario', 'region', 'gas', 'sector', 'unit')
    - years: [2005:5:2060,2070:10:2100]

    - as saved CSV file:
        - concordia\data\compare_wfd_inputs\model.csv

- ``hist`` (pd.Dataframe)
    - line of code: 
        - ``concat([hist_ceds, hist_global, hist_gfed]) ...``
        - input files: 
            - settings.history_path / "ceds_2017_extended.csv"
            - settings.history_path / "gfed/GFED2015_extended.csv"
            - settings.history_path / "global_trajectories.xlsx"

    - MultiIndex: ('country', 'gas', 'sector', 'unit')
    - years: [1750:2015,2020]

    - as saved CSV file:
        - concordia\data\compare_wfd_inputs\hist.csv

- ``gdp`` (pd.Dataframe)
    - line of code: 
        - ``pd.read_excel(settings.scenario_path / "harmonization_overrides.xlsx",index_col=list(range(3)) ...``
        - input file: harmonization_overrides.xlsx 
    
    - MultiIndex: ('model', 'scenario', 'country')
    - years: [2000:5:2100]

    - as saved CSV file:
        - concordia\data\compare_wfd_inputs\gdp.csv

- ``regionmapping.filter(gdp.pix.unique("country"))`` (concordia.RegionMapping)
    - line of code: 
        - ``for model, kwargs in settings.regionmappings.items(): regionmapping = RegionMapping.from_regiondef(**kwargs) ...``
        - input CSV file: 'C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/scenarios/regionmappingH12.csv'

    - type: ``concordia.utils.RegionMapping``
    - data = country
    - name = region
    - Length = 183

    - as saved CSV file:
        - concordia\data\compare_wfd_inputs\regionmapping.csv

- ``indexraster`` (ptolemy.IndexRaster)
    - line of code: 
        - ``IndexRaster.from_netcdf(settings.gridding_path / "ssp_comb_indexraster.nc",chunks={})``
        - input netCDF file: settings.gridding_path / "ssp_comb_indexraster.nc"

    - as saved txt file:
        - concordia\data\compare_wfd_inputs\indexraster_country_info.txt

- ``indexraster_region`` (ptolemy.IndexRaster)
    - line of code: 
        - ``indexraster.dissolve(regionmapping.filter(indexraster.index).data.rename("country")).persist()``
        - input netCDF file: settings.gridding_path / "ssp_comb_indexraster.nc"

    - as saved txt file:
        - concordia\data\compare_wfd_inputs\indexraster_region_info.txt

- ``variabledefs`` (concordia.VariableDefinitions; variabledefs.data: pandas.DataFrame)
    - line of code: 
        - variabledefs = VariableDefinitions.from_csv(settings.variabledefs_path)
        - input CSV file: 'concordia/data/variabledefs-rescue.csv'

    - .data
        - MultiIndex: ('gas', 'sector')

    - as saved CSV file:
        - concordia\data\compare_wfd_inputs\variabledefs.csv

- ``harm_overrides`` (pd.Series[str])
    - line of code: 
        - ``pd.read_excel(settings.scenario_path / "harmonization_overrides.xlsx",index_col=list(range(3))).method``
        - input CSV file: settings.scenario_path / "harmonization_overrides.xlsx"

    - as saved CSV file:
        - concordia\data\compare_wfd_inputs\harm_overrides.csv

- ``settings`` (concordia.settings.Settings)
    - line of code:
        - settings = Settings.from_configconfig_path="config-rescue.yaml", local_config_path="local-config-rescue.yaml", version="TEST-2025-04-10")
    
    - as saved txt file:
        - concordia\data\compare_wfd_inputs\settings.txt

N.B. saving the files of a WorkflowDriver object can be done like ``workflow.save_info(path = your_path)``


WorkflowDriver.grid [concordia.workflow.WorkflowDriver.grid] steps (RESCUE):

-  harmonize_and_downscale()
    - ...

    - ...

- grid_proxy(output_variable, downscaled)
    - ...

    - ...


- verify_and_save()
    - ...

    - ...




CMIP7 Documentation
-------------------

Getting input formats of CMIP7 data aligned with RESCUE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Steps to follow
~~~~~~~~~~~~~~~

1. Configuration file: create the right configuration file settings
2. Proxy files: create all required gridding files
    2.1. CEDS proxy files (CEDS, BB4CMIP):
        2.1.1. CEDS. CEDS documentation of CMIP7 release: https://zenodo.org/records/15001544; README.txt
            2.1.1.1. download from ESGF:
                - link: ...
            2.1.1.2. download from CEDS Zenodo:
                2.1.1.2.1. country masks:
                    - CEDS2019_06_18: https://zenodo.org/records/3249323
                2.1.1.2.2. proxies:
                    - CEDS2019_06_18: https://zenodo.org/records/3249323
                    - Questions: 
                        - where is population.R from?
                        - where to find updates?
                2.1.1.2.3. seasonality:
                    - Questions:
                        - where is seasonality data from?? (The gridded emissions incorporate a monthly seasonal cycle by sector drawing largely from the ECLIPSE project, Carbon Tracker, and EDGAR.)
            2.1.1.3. Download from CEDS GitHub:
                - https://github.com/JGCRI/CEDS/blob/master/input/gridding/gridding_mappings/country_location_index_05.csv
        
        2.1.2. BB4CMIP: 
            2.1.2.1. download from ESGF
                - Questions: 
        
        2.1.3. adjust proxy mapping files (CMIP6: https://zenodo.org/records/2538194 ; CMIP7 files TBD - need input from Steve/Rachel/Hamza )
            2.1.3.1. proxy-CEDS9/proxy_mapping_CEDS9.csv
                2.1.3.1.1. proxy generally is directly CEDS based

                2.1.3.1.2. backup is generally population (also provided by CEDS)
            
            2.1.3.2. seasonality-CEDS9/seasonality_mapping_CEDS9.csv
                2.1.3.2.1. provided by CEDS; e.g. 
        
    b. non-CEDS proxy files (international shipping, CDR)
        i. Shipping: use MariTEAM data
            I. Questions:
                - should we use CEDS? In the RESCUE code it says that there were issues. Will it be a probelm 
        ii. CDR: use RESCUE data
    c. run proxy-generation scripts:
        i. python notebooks/gridding_data/generate_ceds_proxy_netcdfs.py
        ii. python notebooks/gridding_data/generate_non_ceds_proxy_netcdfs.py
3. Historical data (national): process historical emissions in https://github.com/iiasa/emissions_harmonization_historical
    a. currently, this means running the notebooks 0101-0111 and 0201-0203.
    b. the input files are:
        i. cmip7_history_0012.csv
4. Download IAM scenario (regional)
    a. download
    b. place them in the right folder
5. Run workflow
    a. script: notebooks/workflow-cmip7.py (can be loaded as jupyter notebook)
    b. steps:
        i. load and process IAM data
        ii. load and process historical (national) data
        iii. harmonize
        iv. grid (using aneris). Input files for aneris:
            I. ...GDP...
            II. ...POPULATION...
            III. where to choose?
            IV. discuss default methods...
6. {analyse the results}

Configuration file
~~~~~~~~~~~~~~~~~~

File:

- ./notebooks/config_cmip7_v0_testing.yaml

Describes 

- data_path (folder with enough storage space; needs to be specific to the user)
- variabledefs_path [check whether/where it is used] (likely a subfolder of data_path) 
- history_path [check whether/where it is used] (likely a subfolder of data_path)
- regionmappings_path [check whether/where it is used] (likely a subfolder of data_path)
- gridding_path [check whether/where it is used] (likely a subfolder of data_path)
- proxy_path [check whether/where it is used] (likely a subfolder of data_path)
- postprocess_path [check whether/where it is used] (likely a subfolder of data_path)
- variable_template [describe] (string like "" used for ....): "CMIP7 Harmonization|Emissions|{gas}|{sector}"
- base_year: year that used to harmonize in (int): 2023
- region_mappings
    - list of model names participating in CMIP7
    - each with:
        - path: path to country-to-region mapping
        - country_column: the column of the column where strings are expected be in iso3c format
        - region_column: 
        - sep: put to "," as it expects a comma-delimited file format
- country_combinations: as CEDS is the most important historical emissions database for CMIP7, we aim to follow its country/territory split as much as possible. However, other databases (such as SSP data for population and GDP) which is used for creating future grids may miss data for some countries/territories, or may have them in more aggregate format. Therefore, if this is the case, we aggregate the countries/territories of CEDS and handle them as a combination of countries. These can be specified as:
    - new_iso: [list of ceds-countries/territories]
- luc_sectors: list the four BB4CMIP burning sectors + CEDS Agriculture sector
- encoding (do not change)
- ftp [check whether/where it is used; delete if not used]

Input data (files)
~~~~~~~~~~~~~~~~~~


Pre-processing
^^^^^^^^^^^^^^

Creating IAM-region history
"""""""""""""""""""""""""""

Creating gridding files from a CEDS release
"""""""""""""""""""""""""""""""""""""""""""

Settings from a configuration file, uses:

- config_cmip7_v0_testing.yaml
    - gridding_path
    - country_combinations
    - encoding (for encoding of data array to netCDF)

Template format file () from RESCUE:

- ...


Country-masks from CEDS:

- ...

Year files from CEDS:

- ...

Seasonality files from CEDS:

- ...


Creating gridding files from a BB4CMIP release
""""""""""""""""""""""""""""""""""""""""""""""


IAM-region history
^^^^^^^^^^^^^^^^^^

IAM-region scenarios
^^^^^^^^^^^^^^^^^^^^

Input data, files:
- gridding
- historical
- scenario
- 

Input data, location:
-  {LOCAL_PATH}/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/{VERSION}/input

Output data, files:
- ...

Output data, location:
- ...

Notes-to-self
-------------

Model read-in cell is not very clear currently, and breaks when formatting is wrong, without a clear error message.
