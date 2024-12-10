

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

1: Add spatial-proxy generating code (e.g., in
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
implementaiton for these different processes lives in
[`aneris`](https://github.com/iiasa/aneris).


Applications
==========================

Documentation
-------------

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


To-do list
-------------

*The list below is ordered.*
- [ ] update input data: map from ssp_submission downloaded data to concordia input data
- [ ] update variabledefs-cmip7.csv
- [ ] update CEDS data (still harmonization in 2020)
- [ ] use interpolated input files, and move harmonization to 2022
- [ ] run pipeline on a newly submitted MESSAGE ScenarioMIP scenario (without aviation) 
- [ ] update gridding files with new CEDS data (from ESGF, or direct download?)
- [ ] update to BB4CMIP7 national GFED data (from emissions_harmonization_historical?)
- [ ] new SSP data
- [ ] is `country_combinations` still needed?
- [ ] interpolate IAM emissions data
- [ ] 2022 as `base_year`
- [ ] create mapping file with regionmapping following ssp_submission scenario explorer mapping style, using common-definitions / nomenclature
- [ ] register multiple models
- [ ] try new harmonization algorithm 
- [ ] remove or keep alkalinity option? 