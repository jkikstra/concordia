`concordia` - Harmonization, downscaling and gridding of emissions data
=======================================================================

Getting Started
---------------

We suggest using environments to install this package, and in particular using
[`mamba`](https://mamba.readthedocs.io/en/latest/) for the fastest installation
experience (though `conda` and `pip install` work fine too).

mamba/conda
***********

.. code-block:: bash
    $ mamba env create -f environment.yml

pip
***

.. code-block:: bash
    $ pip install -e .[lint,rescue,test]

You can then check to make sure your install is operating as expected

.. code-block:: bash

    pytest tests

Jupyter kernel
**************

The CMIP7 workflow notebooks are executed through papermill, which uses the kernel named in each
notebook's jupytext front matter. Register the environment as a kernel named ``concordia`` once
per machine, or those runs will execute against whichever generic ``python3`` kernel happens to
be registered:

.. code-block:: bash

    python -m ipykernel install --user --name concordia

Projects
--------

This repository hosts several projects that share the harmonization, downscaling and gridding
machinery in ``src/concordia``. Each has its own README with run instructions:

* **CMIP7 ScenarioMIP** — gridded emissions for the CMIP7 fast-track (2022-2100) and extensions
  (2100-2500). This is the active project.
  See `notebooks/cmip7/README.md <notebooks/cmip7/README.md>`_.
* **RESCUE** — the earlier project this tooling was first built for.
  See `notebooks/rescue/README.md <notebooks/rescue/README.md>`_.
* **Generic / cross-project notebooks** — see `notebooks/README.rst <notebooks/README.rst>`_.

Running
-------

Start from the README of the project you are working on, linked above. There is no single
entrypoint: each project has its own configuration files, input-data requirements and run order.

For CMIP7 specifically, begin with `notebooks/cmip7/README.md <notebooks/cmip7/README.md>`_ and
the input-data requirements in
`notebooks/cmip7/docs/inputs.md <notebooks/cmip7/docs/inputs.md>`_.

License
-------

Licensed under Apache 2.0. See the LICENSE file for more information
