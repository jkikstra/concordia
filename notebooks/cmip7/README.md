# CMIP7 workflow for emissions gridding

This describes how to produce gridded emissions for the CMIP7 ScenarioMIP experiments: a **fast-track** pipeline
(producing data for 2022–2100) followed by an **extensions** pipeline (producing data for 2105–2500) that continues from the fast-track
output. All files are in this folder unless noted otherwise; scripts are jupytext-paired
`.py`/`.ipynb` — edit the `.py`.

Some paths in the scripts are currently hardwired and may need manual adaptation if re-running.

**Starting point:** pre-harmonised scenario emissions data, prepared in <https://github.com/iiasa/emissions_harmonization_historical>.
This repository does contain harmonisation functionality — used for the RESCUE project — but it
is not used for CMIP7.

## Before you run anything

| You need to know | Read |
| --- | --- |
| What must exist on disk, and where each input comes from | [`docs/inputs.md`](docs/inputs.md) |
| Where to download a version's input bundle, and who to ask | [`docs/versions.md`](docs/versions.md) |
| What each config key means | comments at the top of `config_cmip7_v0-4-0.yaml` |
| How to validate and publish finished output | [`docs/esgf-upload.md`](docs/esgf-upload.md) |
| What changed between published versions | the Zenodo record — see below |

The two workflow scripts run through papermill and pin the jupytext kernel to `concordia`.
Register it once per machine:

```bash
python -m ipykernel install --user --name concordia
```

## Fast-track pipeline (2022–2100)

Final config for CMIP7: `config_cmip7_v0-4-0.yaml`

### 1. Prepare proxy files

Run once per config version, in this order where a dependency applies:

1. `prep_countrymask-from-ceds.py` — builds the country index raster
   (`ssp_comb_indexraster_splitsudankosovopalestine.nc`) used throughout both pipelines to
   aggregate gridded data to country level.
2. `prep_proxyfuture-anthro-from-ceds-cmip7-esgf.py` — spatial proxies for anthropogenic,
   shipping, and aircraft emissions, from CEDS gridded historical emission files available via ESGF.
3. `prep_proxyfuture-openburning-from-dres-cmip7-esgf.py` — spatial proxies for openburning
   emissions, from BB4CMIP7 historical openburning emissions files available via ESGF.
4. `prep_proxyfuture-cdr-from-rescue.py` — CDR proxy (`CDR_CO2.nc`).
5. `prep_proxyfuture-cdr-erw.py` — Enhanced Weathering CO2 proxy; reads the `CDR_CO2.nc`
   template from step 4.
6. `prep_h2openburning_foresttypespergridcell.py` — H2/CO emission-factor proxy
   (`EF_h2_div_EF_co.nc`), used to derive H2 openburning emissions from CO.
7. `prep_proxyfuture-anthro-from-ceds-cmip7-esgf-VOCspeciation.py` — VOC speciation share
   proxies for anthropogenic emissions.
8. `prep_proxyfuture-openburning-from-dres-cmip7-esgf-VOCspeciation.py` — VOC speciation share
   proxies for openburning emissions.

### 2. Workflow

- `workflow_cmip7-fast-track.py` — harmonises, downscales, and grids one marker scenario at a
  time: main species (anthro, AIR-anthro, openburning), H2 openburning, and VOC speciation
  (anthro + openburning). This happens in two steps: downscaling to country-level, using the country index raster,
  then downscaling to the gridded level using the spatial proxies. Includes subsequent spatial harmonisation of patterns
  against CEDS 2023, as well as timeseries corrections and a number of built-in checks for QC.
  Run via `scripts/cmip7/driver_workflow_cmip7-fast-track.py` (papermill) for one or more markers.

### 3. Checking

- `check_gridded_scenario_qc.py` — the main QC tool: file inventory, min/max sanity checks,
  downscaled-data QC, annual totals compared across input/harmonised/gridded, sectoral totals,
  animated grid maps, documentation plots, and per-location timeseries vs. CEDS/BB4CMIP7 history.
  Modules can be toggled on/off.
- `check_VOCspeciation_share_proxies.py` — validates the VOC-speciation share proxy files
  (output of steps 1.7 and 1.8 above) before they're used for gridding.

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

## Outputs

Everything lands under `{out_path}/results/{GRIDDING_VERSION}/`, where `GRIDDING_VERSION` is
`{marker}_{VERSION_ESGF}` for the fast-track and `{marker}-ext_{VERSION_ESGF}` for the extensions.
`out_path` is set in the config — note it is one of the keys declared more than once, so check the
effective value before hunting for missing files.

| Output | What it is |
| --- | --- |
| `harmonization-{version}.csv` | Harmonised IAMC-format data |
| `downscaled-only-{version}.csv` | Country-level downscaled data. Also the **history input to the extensions pipeline** |
| `{gas}-em-{sector}_{FILE_NAME_ENDING}.nc` | The gridded NetCDFs — main species, H2, and VOC speciation |
| `areacella_*.nc` | Grid-cell area file with updated metadata |
| `plots/` | PNG maps and timeseries comparison plots |
| `check_annual_totals/`, `check_annual_totals_ext/`, `check_NMVOC_sums/`, `check_VOC_sums/` | QC CSVs and plots from the built-in checks |
| `debug_{version}.log` | Run log |

## Other tools

- `compare_gridded_versions.py` — generic diff between any two gridded output folders
  (exact-equality checks + attribute diffs); not scenario-specific. Run via
  `scripts/cmip7/driver_compare_gridded_versions.py`.

## Naming conventions

Filenames in this folder follow `{type-of-file}_{description-of-purpose-or-action}`, and the
prefix tells you what a script is for. This is also what decides which folder a script lives in.

| Prefix | Meaning |
| --- | --- |
| `config_*` | Main configuration file; can serve any type of script |
| `prep_*` | Runs **before** `workflow_*`; builds proxies and masks (includes `prep_proxy_*`) |
| `workflow_*` | Input (harmonised IAM emissions) → downscaled and gridded data products |
| `check_*` | Checks on data produced by `prep_*` and `workflow_*`; numerical or visual |
| `investigate_*` | Looks into input files. Produces nothing the workflow needs, and analyses no workflow output — hence `investigate/` |

Two conventions worth stating explicitly:

- A `{project-name}` folder is only needed when a script is *not* reusable unchanged across
  projects. Generic scripts stay in the root `notebooks/` folder. Moving a script back out to the
  root later, once it has been generalised, is expected.
- Avoid `{version}` in filenames. Add one only when multiple versions genuinely must coexist —
  for example to run the same workflow under several configurations. Never end a filename with `_`.

RESCUE-project filenames predate CMIP7 and deliberately do not follow this structure.

A `workflow-postprocess_*` prefix also exists in `archive/`: it did additional processing on grid
files after `workflow_*`. That work is now folded into the workflow scripts themselves, so the
prefix should not be used for new scripts.

## Known issues in the fast-track workflow

Tracked upstream, in the issue tracker the team uses for CMIP7 work:

- [IAMconsortium/concordia#94](https://github.com/IAMconsortium/concordia/issues/94) — the five
  open bugs below, with line numbers and suggested fixes.
- [IAMconsortium/concordia#87](https://github.com/IAMconsortium/concordia/issues/87) — the
  improvement backlog for this file (code quality, performance, robustness, structure).

Five bugs are open, and two of them will crash a legitimate run:

| | Effect |
| --- | --- |
| **B4** | `assert remainder_diff_2023 < 50` has no `abs()`, so a scenario much *larger* than the CEDS reference passes silently |
| **B5** | `NameError` when `run_main_gridding=False` and `run_openburning_h2=True` |
| **B6** | `check_harmonization_consistency` called twice identically; doubles that stage's runtime |
| **B7** | `new_stem` read from a loop variable after the loop, in three places |
| **B8** | `_what_emissions_variable_type` raises `UnboundLocalError` for an unclassifiable file |

Check those issues before debugging a failed run — the failure may already be known.

## Other folders

- `archive/` — scripts superseded by the current workflow, or tied to old config versions; kept
  for reference, not part of the live pipeline.
- `investigate/` — exploratory notebooks that don't feed or check the workflow.
- `docs/` — input-data layout, version/data locations and contacts, and the ESGF publishing
  procedure.

## Further documentation and published data

**<https://zenodo.org/records/19730076>** — the published datasets and the canonical record of
what changed between versions. This repository does not duplicate that changelog.

Contacts and per-version input-data locations: [`docs/versions.md`](docs/versions.md).
