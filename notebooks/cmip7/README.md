# CMIP7 workflow for emissions gridding

This describes how to produce emissions grids for CMIP7 ScenarioMIP: a **fast-track** pipeline
(producing data for 2022–2100) followed by an **extensions** pipeline (producing data for 2105–2500) that continues from the fast-track
output. All files are in this folder unless noted otherwise; scripts are jupytext-paired
`.py`/`.ipynb` — edit the `.py`.

**Starting point:** a pre-harmonised scenario emissions data is assumed as input. Harmonisation itself happens
upstream, in `iiasa/emissions_harmonization_historical`.

## Fast-track pipeline (2022–2100)

Config: `config_cmip7_v0-4-0.yaml`

### 1. Pre-processing

Run once per config version, in this order where a dependency applies:

1. `prep_countrymask-from-ceds.py` — builds the country index raster
   (`ssp_comb_indexraster_splitsudankosovopalestine.nc`) used throughout both pipelines to
   aggregate gridded data to country level.
2. `prep_proxyfuture-anthro-from-ceds-cmip7-esgf.py` — spatial proxies for anthropogenic,
   shipping, and aircraft emissions, from ESGF CEDS files.
3. `prep_proxyfuture-openburning-from-dres-cmip7-esgf.py` — spatial proxies for openburning
   emissions, from ESGF BB4CMIP7 files.
4. `prep_proxyfuture-cdr-from-rescue.py` — CDR proxy (`CDR_CO2.nc`).
5. `prep_proxyfuture-cdr-erw.py` — Enhanced Weathering CO2 proxy; reads the `CDR_CO2.nc`
   template from step 4, so run after it.
6. `prep_h2openburning_foresttypespergridcell.py` — H2/CO emission-factor proxy
   (`EF_h2_div_EF_co.nc`), used to derive H2 openburning emissions from CO.
7. `prep_proxyfuture-anthro-from-ceds-cmip7-esgf-VOCspeciation.py` — VOC speciation share
   proxies for anthropogenic emissions.
8. `prep_proxyfuture-openburning-from-dres-cmip7-esgf-VOCspeciation.py` — VOC speciation share
   proxies for openburning emissions.

### 2. Workflow

- `workflow_cmip7-fast-track.py` — harmonises, downscales, and grids one scenario marker at a
  time: main species (anthro, AIR-anthro, openburning), H2 openburning, and VOC speciation
  (anthro + openburning). Includes built-in QC (spatial harmonisation vs. CEDS 2023, timeseries
  corrections, VOC-sum checks). Run via `scripts/cmip7/driver_workflow_cmip7-fast-track.py`
  (papermill) for one or more markers.

### 3. Checking

- `check_gridded_scenario_qc.py` — the main QC tool: file inventory, min/max sanity checks,
  downscaled-data QC, annual totals compared across input/harmonised/gridded, sectoral totals,
  animated grid maps, documentation plots, and per-location timeseries vs. CEDS/BB4CMIP7 history.
  Modules can be toggled on/off.
- `check_gridded-scenarios-compare-to-ceds-esgf.py` — detailed sector-by-sector comparison of
  gridded output to CEDS reference grids in the harmonisation year, plus timeseries plots for
  specific points/areas (slow).
- `check_plot-global-total-timeseries.py` — simple global annual-total timeseries plots per
  species/sector, with historical data overlaid.
- `check_VOCspeciation_share_proxies.py` — validates the VOC-speciation share proxy files
  (output of steps 7–8 above) before they're used for gridding.

## Extensions pipeline (2105–2500)

Config: `config_cmip7_v0-4-0-EXT.yaml`

**Requires the fast-track pipeline to have already been run for the same scenario/version** —
this pipeline reads fast-track's `downscaled-only-*.csv` as its history input, and fast-track's
final gridded NetCDFs to anchor its 2100 boundary correction.

### 1. Pre-processing

1. `prep_extensions_gdp.py` — extends the GDP proxy beyond 2100.
2. `prep_proxyfuture-extensions.py` — freezes the 2100 spatial pattern (from the fast-track
   proxies and, where available, the scenario's own 2100 gridded output) and repeats it across
   all extension years, producing everything under `proxy_rasters_extensions/`.
3. `prep_downscaled_to_country_from_gridded.py` — only if needed: re-derives the
   `downscaled-only-*.csv` history file from the final fast-track gridded NetCDFs, for cases
   where post-gridding fixes have made the original CSV stale.

### 2. Workflow

- `workflow_cmip7-extensions.py` — harmonises, downscales, and grids one scenario marker at a
  time for 2100–2500, then anchors the result to fast-track's 2100 values via a fading additive
  correction (the 2100 timestep is dropped from the output; fast-track owns 2100). Includes the
  same built-in QC/VOC-sum checks as fast-track. Run via
  `scripts/cmip7/driver_workflow_cmip7_extensions.py`.

### 3. Checking

- `check_gridded_scenario_qc-ext.py` — same QC modules as `check_gridded_scenario_qc.py`, for
  extension output.
- `check_gridded_scenario_junctions-ext.py` — verifies continuity across the three segments
  (CEDS historical 2000–2023, fast-track 2022–2100, extension 2105–2500) at their boundaries.

## Other tools

- `compare_gridded_versions.py` — generic diff between any two gridded output folders
  (exact-equality checks + attribute diffs); not scenario-specific. Run via
  `scripts/cmip7/driver_compare_gridded_versions.py`.

## Other folders

- `archive/` — scripts superseded by the current workflow, or tied to old config versions; kept
  for reference, not part of the live pipeline.
- `investigate/` — exploratory notebooks that don't feed or check the workflow.
- `untracked/` — personal work-in-progress scripts (gitignored).
