# Input data for the CMIP7 gridding workflow

What must exist on disk before any `prep_` or `workflow_` script will run, and where each piece
comes from.

Paths below are given relative to the keys set in the config YAML
(`config_cmip7_v0-4-0.yaml` for the fast-track, `config_cmip7_v0-4-0-EXT.yaml` for the
extensions). Each key is documented in comments inside those files; this document covers the
*contents* those keys must point at.

For where to download a given version's input bundle, and who to ask, see
[`versions.md`](versions.md).

---

## Starting point

The workflow assumes you already have a **pre-harmonised scenario**.

This repository does contain harmonisation functionality — it was used for the RESCUE project —
but it is not used for CMIP7. CMIP7 scenarios are harmonised upstream in
[`iiasa/emissions_harmonization_historical`](https://github.com/iiasa/emissions_harmonization_historical)
and arrive here already harmonised.

---

## `gridding_path`

Hosts the raw inputs from which downscaling proxies are derived. Most of these must be downloaded
before you start.

```
gridding_path/
├── esgf/                          # downloaded from ESGF
│   ├── bb4cmip7/                  # raw BB4CMIP7 open-burning emissions
│   └── ceds/
│       ├── CMIP7_AIR/             # raw CEDS aircraft emissions
│       └── CMIP7_anthro/          # raw CEDS other anthropogenic emissions
├── iiasa/
│   └── cdr/
│       ├── rescue/                # CDR proxies from the RESCUE project — only CDR_CO2.nc is used
│       └── pratama_joshi/         # enhanced-weathering proxy (Yoga Pratama, Siddharth Joshi)
├── proxy_rasters/                 # CREATED by the prep_ steps — do not populate by hand
├── example_files/                 # one RESCUE file, read only to pick up the target resolution
├── iso_mask/
│   └── eez_v12.gpkg               # exclusive economic zone raster
├── country_location_index_05.csv  # country coordinates, from the RESCUE project
└── areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc
                                   # grid-cell area on the CEDS grid; available from ESGF, kept
                                   # here because it is the common grid for all current scenario work
```

`proxy_rasters/` is an output of the pre-processing stage, not an input. It appears here because
the workflow reads it back from the same root.

## `history_path`

Country-level CEDS + GFED history.

Needs a file such as `cmip7_history_countrylevel_251024.csv`, produced in
[`emissions_harmonization_historical`](https://github.com/iiasa/emissions_harmonization_historical)
as
[`COUNTRY_LEVEL_HISTORY`](https://github.com/iiasa/emissions_harmonization_historical/blob/28e6d69991205b3a824936538ec62358480d80ed/src/emissions_harmonization_historical/constants_5000.py#L126).

> **Important:** this must be exactly the same history used in the harmonisation step upstream
> ([`5094_harmonisation.py`](https://github.com/iiasa/emissions_harmonization_historical/blob/main/notebooks/5094_harmonisation.py)).
> A mismatch here produces a silent inconsistency between the harmonised scenario and the
> gridding baseline, not an error.

## `regionmappings_path` / `regionmappings`

For each IAM, which countries belong to which model region.

Defined in
[`common-definitions`](https://github.com/IAMconsortium/common-definitions/tree/main/definitions/region/native_regions)
and produced by
[`5010_create-region-mapping.py`](https://github.com/iiasa/emissions_harmonization_historical/blob/main/notebooks/5010_create-region-mapping.py),
with a few local additions for minor territories not yet in common-definitions.

> **Known sharp edge:** model names (including version numbers) are currently hard-coded. When a
> model version changes upstream, check that the mapping still resolves.

## `scenario_path`

- The IAM scenario(s) to downscale, as produced by
  [`extract-emissions-results.py`](https://github.com/iiasa/emissions_harmonization_historical/blob/28e6d69991205b3a824936538ec62358480d80ed/scripts/extract-emissions-results.py#L30)
- `ssp_basic_drivers_release_3.2.beta_full_gdp.csv` — country-level GDP projections by SSP
- `harmonization_overrides.xlsx` — an empty file that could host harmonisation method overrides.
  Not used for CMIP7; kept empty so the read succeeds.

## `variabledefs_path`

A CSV specifying every species–sector combination that gets produced: which method is used
(global or country), which proxy file and proxy variable, and which output variable/file the
result lands in (column `output_var`).

This file drives what the workflow produces. If a species or sector is missing from the output,
check here first.

## `proxy_path`

Spatial proxies for gridding. Produced by the pre-processing (`prep_`) step — see the run order
in [`../README.md`](../README.md). Not something you download.

---

## Extensions inputs

The extensions pipeline (2100–2500) additionally requires that the **fast-track has already been
run for the same marker**, because it anchors on fast-track output:

- `downscaled-only-{marker}_{VERSION_ESGF}.csv` — country-level downscaled fast-track result
- the final gridded fast-track NetCDFs, read for the 2100 anchor year

If the downscaled CSV is stale relative to the gridded files, regenerate it with
`prep_downscaled_to_country_from_gridded.py` rather than re-running the whole fast-track.
