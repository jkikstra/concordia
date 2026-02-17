# CMIP7 ScenarioMIP Emissions Workflow Documentation

> **Script:** `workflow_cmip7-fast-track.py`
> **Format:** Jupytext percent-format (`.py` ↔ `.ipynb`)
> **Constraint:** Runs one scenario at a time.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration Parameters](#2-configuration-parameters)
3. [Step-by-Step Workflow](#3-step-by-step-workflow)
   - [Step 1: Settings & Imports](#step-1-settings--imports)
   - [Step 2: Read Settings from YAML](#step-2-read-settings-from-yaml)
   - [Step 3: Read Variable Definitions](#step-3-read-variable-definitions)
   - [Step 4: Read Region Definitions](#step-4-read-region-definitions)
   - [Step 5: Read & Process Historical Emissions](#step-5-read--process-historical-emissions)
   - [Step 6: Read & Process IAM Scenario Data](#step-6-read--process-iam-scenario-data)
   - [Step 7: Read Harmonization Overrides](#step-7-read-harmonization-overrides)
   - [Step 8: Prepare GDP Proxy](#step-8-prepare-gdp-proxy)
   - [Step 9: Country & Sector Coverage Checks](#step-9-country--sector-coverage-checks)
   - [Step 10: Build the WorkflowDriver](#step-10-build-the-workflowdriver)
   - [Step 11: Harmonization & Downscaling](#step-11-harmonization--downscaling)
   - [Step 12: Quality Control on Downscaled Data](#step-12-quality-control-on-downscaled-data)
   - [Step 13: Gridding to NetCDF](#step-13-gridding-to-netcdf)
   - [Step 14: Spatial Harmonization](#step-14-spatial-harmonization)
   - [Step 15: Timeseries Corrections](#step-15-timeseries-corrections)
   - [Step 16: H₂ Openburning Derivation](#step-16-h₂-openburning-derivation)
   - [Step 17: VOC Speciation](#step-17-voc-speciation)
   - [Step 18: Post-processing & Plotting](#step-18-post-processing--plotting)
4. [Debug Patches](#4-debug-patches)
5. [Output Files](#5-output-files)
6. [Data Flow Diagram](#6-data-flow-diagram)

---

## 1. Overview

This workflow takes Integrated Assessment Model (IAM) emissions scenarios through a complete processing pipeline to produce gridded NetCDF files suitable for CMIP7 climate models. The high-level pipeline is:

```
IAM scenarios ──► Harmonization ──► Downscaling ──► Gridding ──► Spatial Harmonization
                                                                      │
                                                    ◄────────────────┘
                                               Timeseries Corrections
                                                      │
                                              VOC Speciation / H₂
                                                      │
                                              Quality Control & Plots
```

### Key Libraries

| Library | Role |
|---------|------|
| **concordia** | Core workflow: `WorkflowDriver`, `RegionMapping`, `VariableDefinitions`, `Settings` |
| **aneris** | Harmonization & downscaling algorithms |
| **ptolemy** | Raster-based spatial gridding (`IndexRaster`) |
| **pandas_indexing** | MultiIndex manipulation (`isin`, `ismatch`, `semijoin`, `extractlevel`) |
| **xarray / dask** | NetCDF I/O and parallel computation |

---

## 2. Configuration Parameters

All parameters are set at the top of the script in the `tags=["parameters"]` cell, making them injectable by **papermill** for batch runs.

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `HISTORY_FILE` | `str` | `"country-history_...csv"` | Filename of the country-level historical emissions CSV inside `history_path`. |
| `SETTINGS_FILE` | `str` | `"config_cmip7_v0-4-0.yaml"` | YAML configuration file (relative to `notebooks/cmip7/`). |
| `VERSION_ESGF` | `str` | `"1-1-0-alpha"` | Version string used in ESGF-formatted output filenames. |
| `marker_to_run` | `str` | `"vl"` | Which marker scenario to process. Options: **`h`**, **`hl`**, **`m`**, **`ml`**, **`l`**, **`ln`**, **`vl`**. |
| `GRIDDING_VERSION` | `str \| None` | `"{marker}_{VERSION_ESGF}"` | Subdirectory name under `out_path` for this run's outputs. Defaults to `"{marker_to_run}"` if `None`. |

### Workflow Stage Toggles

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `run_main` | `bool` | `True` | Run harmonization, downscaling, and data export. If `False`, only supplemental workflows run. |
| `run_main_gridding` | `bool` | `True` | Produce gridded NetCDF files (the most time-consuming step). |
| `SKIP_EXISTING_MAIN_WORKFLOW_FILES` | `bool` | `False` | If `True`, skip gridding for files already on disk. |
| `run_spatial_harmonisation` | `bool` | `True` | Apply spatial harmonization against raw CEDS anthro data (requires local CEDS files). |
| `run_anthro_timeseries_correction` | `bool` | `True` | Apply global-total timeseries corrections to anthropogenic grids. |
| `run_AIR_anthro_timeseries_correction` | `bool` | `True` | Apply timeseries corrections to aviation (AIR) grids. |
| `run_openburning_timeseries_correction` | `bool` | `True` | Apply timeseries corrections to openburning grids. |
| `run_openburning_h2` | `bool` | `True` | Derive H₂ openburning emissions from CO × emission-factor ratios. |
| `run_anthro_supplemental_voc` | `bool` | `True` | Run VOC speciation for anthropogenic (CEDS) sectors. |
| `run_openburning_supplemental_voc` | `bool` | `True` | Run VOC speciation for openburning (BB4CMIP7) sectors. |

### Species & Sector Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DO_GRIDDING_ONLY_FOR_THESE_SPECIES` | `list[str] \| None` | `["CO"]` | If set, restrict all processing to these gas species (e.g., `["CO2", "SO2"]`). `None` = all. |
| `DO_GRIDDING_ONLY_FOR_THESE_SECTORS` | `list[str] \| None` | `None` | If set, restrict processing to these emission types: `"anthro"`, `"openburning"`, `"AIR_anthro"`. |
| `DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES` | `list[str] \| None` | `None` | Filter anthro VOC speciation to specific species. |
| `DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES` | `list[str] \| None` | `None` | Filter openburning VOC speciation to specific species. |

### Post-processing / Plotting Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot_timeseries` | `bool` | `True` | Generate timeseries comparison plots. |
| `PLOT_GASES` | `list[str] \| None` | `None` (all) | Gases to plot. |
| `PLOT_SECTORS` | `list[str] \| None` | `None` (all) | Sectors to plot. |
| `ceds_data_location` | `Path \| None` | `None` → `settings.postprocess_path / "CMIP7_anthro"` | Path to raw CEDS anthro NetCDF files for spatial harmonization. |
| `ceds_data_location_voc` | `Path \| None` | `None` → `settings.postprocess_path / "CMIP7_anthro_VOC"` | Path to raw CEDS VOC NetCDF files. |
| `ceds_data_location_AIR` | `Path \| None` | `None` → `settings.postprocess_path / "CMIP7_AIR"` | Path to raw CEDS aviation NetCDF files. |

### Debug Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_harmonization` | `bool` | `True` | **Debug flag.** If `True`, bypass harmonization entirely — use raw model data, then downscale. Used to isolate whether harmonization causes negative values. |

---

## 3. Step-by-Step Workflow

### Step 1: Settings & Imports

All parameters are declared in the first `tags=["parameters"]` cell. This cell is the injection point for **papermill** when batch-running multiple scenarios.

Packages imported include `concordia`, `aneris`, `pandas_indexing`, `xarray`, `dask`, `ptolemy`, `pycountry`, `matplotlib`, and more.

### Step 2: Read Settings from YAML

```
settings = Settings.from_config(
    version=GRIDDING_VERSION,
    local_config_path=Path(HERE, SETTINGS_FILE)
)
```

The YAML file (`config_cmip7_v0-4-0.yaml`) defines:

| Setting | Description |
|---------|-------------|
| `out_path` | Root output directory |
| `data_path` | Root data directory |
| `variabledefs_path` | Path to variable definitions CSV |
| `history_path` | Directory containing historical emissions CSV |
| `scenario_path` | Directory containing harmonized IAM scenario CSVs and overrides |
| `regionmappings_path` | Directory containing IAM region-to-country mapping CSVs |
| `gridding_path` | Path to gridding pattern files (index rasters, proxies) |
| `proxy_path` | Path to proxy rasters for gridding |
| `postprocess_path` | Path to raw CEDS files for spatial harmonization |
| `base_year` | Harmonization base year (currently **2023**) |
| `regionmappings` | Per-model region definition files (7 IAMs configured) |
| `country_combinations` | Country aggregation rules (e.g., `sdn` + `ssd`) |
| `luc_sectors` | Land-use-change sectors treated specially in harmonization |

### Step 3: Read Variable Definitions

```
variabledefs = VariableDefinitions.from_csv(settings.variabledefs_path)
```

The variable definitions CSV specifies for each emission variable:
- `variable` name
- `sector` and `gas` components
- Whether it is **global** (e.g., international shipping) or **regional** (country-level downscaling)
- Gridding proxy to use

If `DO_GRIDDING_ONLY_FOR_THESE_SPECIES` is set, the definitions are filtered to include only the specified gases. Similarly for `DO_GRIDDING_ONLY_FOR_THESE_SECTORS`.

### Step 4: Read Region Definitions

```
regionmappings = {}
for m, kwargs in settings.regionmappings.items():
    regionmapping = RegionMapping.from_regiondef(**kwargs)
    regionmapping.data = regionmapping.data.pix.aggregate(
        country=settings.country_combinations, agg_func="last"
    )
    regionmappings[m] = regionmapping
```

For each IAM, a CSV maps ISO-3166 country codes → IAM regions. Country combinations (e.g., merging South Sudan into Sudan) are applied via `pix.aggregate`.

### Step 5: Read & Process Historical Emissions

**Source:** Country-level historical emissions CSV (from CEDS + GFED/BB4CMIP7).

**Processing steps:**
1. Drop `model`/`scenario` columns, rename `region` → `country`.
2. Extract `gas` and `sector` from the `variable` column using `extractlevel`.
3. Separate into:
   - **Non-global** data: country-level (drop international shipping & aircraft).
   - **Global** data: international shipping & aircraft, renamed to `country="World"`.
   - **Country sum**: sum of all countries for non-shipping/aircraft sectors, assigned `country="World"`.
4. Recombine into a single DataFrame.

**Variable renaming:**
- `Sulfur` → `SO2` (aligns with CEDS/BB4CMIP7 naming)
- `VOC` → `NMVOC` (for anthropogenic sectors)
- `NMVOC` → `NMVOCbulk` (for openburning sectors only, to match BB4CMIP7)

### Step 6: Read & Process IAM Scenario Data

**Source:** `harmonised-gridding_{MODEL}.csv` — already-harmonized scenario data.

**Processing:**
1. Load using `cmip7_utils.load_data()`.
2. Filter to a single scenario using `cmip7_utils.filter_scenario()`.
3. Keep only IAMC columns + year columns from `base_year` to 2100.
4. Extract `gas` and `sector` from the `variable` column.
5. Apply the same Sulfur/VOC renaming as historical data.
6. Save processed data as `scenarios_processed.csv`.

### Step 7: Read Harmonization Overrides

```
harm_overrides = pd.read_excel(
    settings.scenario_path / "harmonization_overrides.xlsx",
    index_col=list(range(3)),
).method
```

The overrides file specifies which harmonization method to use for particular gas/sector/region combinations. In the current CMIP7 pipeline, this file is **empty** because the input scenarios are already harmonized. However, the system supports overrides like `constant_ratio`, `reduce_ratio_2080`, etc.

### Step 8: Prepare GDP Proxy

**Source:** SSP GDP|PPP data from `ssp_basic_drivers_release_3.2.beta_full_gdp.csv`.

**Processing:**
1. Load GDP per SSP scenario, convert country names to ISO-3 codes via `pycountry`.
2. Filter to SSP1–SSP5, select `GDP|PPP` in `billion USD_2017/yr`.
3. Exclude regional aggregates (e.g., `"africa (r10)"`).
4. Interpolate to annual resolution (2021–2100).
5. Merge with 2020 historical GDP for interpolation to 2023/2024.
6. Guess SSP match for each IAM scenario via `cmip7_utils.guess_ssp()`.
7. Join GDP to create `GDP_per_pathway` aligned with the scenario data.

The GDP proxy is used by the **`ipat_2100_gdp`** (intensity convergence) downscaling method for most sectors.

### Step 9: Country & Sector Coverage Checks

- Compute intersection of countries present in: GDP, historical emissions, and region mappings.
- Mayotte (`myt`) and Guam (`gum`) are excluded (missing historical data for some sectors).
- CDR sectors (`Enhanced Weathering`, `BECCS`, `Direct Air Capture`, `Ocean`, `Biochar`, `Soil Carbon Management`, `Other CDR`) are expected to be missing from history — zero-filled rows are added.

### Step 10: Build the WorkflowDriver

```python
workflow = WorkflowDriver(
    model=iam_df,
    hist=hist,
    gdp=GDP_per_pathway,
    regionmapping=regionmapping.filter(countries_with_all_data),
    indexraster_country=indexraster,
    indexraster_region=indexraster_region,
    variabledefs=variabledefs,
    harm_overrides=harm_overrides,
    settings=settings,
)
```

The `WorkflowDriver` is the central orchestrator. It holds:
- Three `GlobalRegional` containers: `harmonized`, `downscaled`, `history_aggregated` — each with `.globallevel`, `.regionlevel`, `.countrylevel` slots.

Workflow info is saved to disk via `workflow.save_info()` for auditability.

### Step 11: Harmonization & Downscaling

```python
downscaled = workflow.harmonize_and_downscale()
```

This is the most complex step in the pipeline. It performs two conceptually distinct operations — **harmonization** (adjusting IAM regional trajectories to match observed history in the base year) followed by **downscaling** (distributing regional totals to individual countries) — and it stores multiple intermediate results on the `workflow` object that are consumed by later steps (export, gridding, plotting).

#### 11.1 The `WorkflowDriver` Internal State

The `WorkflowDriver` object holds three `GlobalRegional` containers that are **populated as side effects** of `harmonize_and_downscale()`:

```
workflow.history_aggregated   ──► GlobalRegional(globallevel, regionlevel, countrylevel)
workflow.harmonized           ──► GlobalRegional(globallevel, regionlevel, countrylevel)
workflow.downscaled           ──► GlobalRegional(globallevel, regionlevel, countrylevel)
```

Each `GlobalRegional` has three slots (`.globallevel`, `.regionlevel`, `.countrylevel`), all initially `None`. They are filled by the three sub-methods described below. A `.data` property concatenates the three slots into a single DataFrame.

These containers are **not just book-keeping** — they are read by later workflow steps:

| Container | Downstream consumer | Purpose |
|-----------|---------------------|---------|
| `workflow.harmonized` | `workflow.harmonized_data` property → export CSV | Produces the `harmonization-{version}.csv` file in IAMC format (unharmonized model + harmonized + history, side by side). |
| `workflow.history_aggregated` | `workflow.harmonized_data` property; `workflow.grid_proxy()` | The aggregated history is prepended to the downscaled time series before gridding (to provide the historical spatial pattern for years before `base_year`). |
| `workflow.downscaled` | `workflow.grid_proxy()`; quality-control checks; `downscaled-only-{version}.csv` export | The country-level data that gets distributed across grid cells. |

The primary inputs to the process are:

| Input | Index levels | Description |
|-------|--------------|-------------|
| `workflow.model` | `(model, scenario, region, gas, sector, unit)` | IAM scenario data at IAM-region level. |
| `workflow.hist` | `(country, gas, sector, unit)` | Historical emissions at country level. |
| `workflow.gdp` | `(ssp, country)` | GDP proxy for intensity-convergence downscaling. |
| `workflow.regionmapping` | `country → region` | Maps ISO-3 country codes to IAM region names. |

#### 11.2 The Three Processing Levels

`harmonize_and_downscale()` calls three sub-methods sequentially and concatenates their return values (country-level DataFrames) into the final `downscaled` result:

```python
concat(skipnone(
    self.harmdown_globallevel(variabledefs),    # (1)
    self.harmdown_regionlevel(variabledefs),    # (2)
    self.harmdown_countrylevel(variabledefs),   # (3)
))
```

Which variables go through which path is determined by `variabledefs`: each variable definition specifies a gridding level (global, region, or country).

##### (1) `harmdown_globallevel()` — International shipping & aircraft

- **Scope:** Variables flagged as `global` in the variable definitions.
- **Input filtering:** `model` rows where `region="World"`; `hist` rows where `country="World"`, renamed to `region`.
- **Harmonization:** `harmonize(model, hist, ...)` adjusts the World-level trajectory.
- **"Downscaling":** There is no spatial disaggregation. The harmonized world total is directly stored as `downscaled.globallevel` with `country="{region}"` (i.e., `country="World"`).
- **Side effects:** Sets `workflow.harmonized.globallevel` and `workflow.history_aggregated.globallevel`.

##### (2) `harmdown_regionlevel()` — Region-level variables

- **Scope:** Variables flagged as `regionlevel`.
- **Input filtering:** `model` rows filtered to mapped regions; `hist` aggregated from country to region using `regionmapping.aggregate()`.
- **Harmonization:** `harmonize(model, hist_agg, ...)` adjusts at IAM-region level.
- **"Downscaling":** No country split. The harmonized region value is stored as `downscaled.regionlevel` with `country="{region}"` (the region name is used as the country label).
- **Side effects:** Sets `workflow.harmonized.regionlevel` and `workflow.history_aggregated.regionlevel`.

##### (3) `harmdown_countrylevel()` — The main downscaling path

This is where the actual spatial disaggregation happens. The logic iterates over **country groups** (groups of countries that share the same set of available proxy weights):

```
For each country_group:
  1. Filter regionmapping to this group's countries
  2. Filter model and hist to this group's variables
  3. Aggregate hist from country→region: hist_agg = regionmapping.aggregate(hist)
  4. HARMONIZE: harm = harmonize(model, hist_agg, ...)    ← adjusts at region level
  5. aggregate_subsectors(harm)                           ← collapse sub-sectors
  6. DOWNSCALE: down = downscale(harm, hist, gdp, ...)    ← distribute to countries
  7. Add zero-rows for missing regions/countries
```

The critical dependency is at step 6: **the `downscale()` function receives the harmonized regional data as its first argument**, not the raw model data. This is because:

- Downscaling distributes a **regional total** to countries.
- That regional total must first be harmonized to match observed history in the base year.
- If the base-year regional total in the model differs from the sum of country-level history, downscaling weights would not sum correctly.

**Side effects:** Sets `workflow.harmonized.countrylevel`, `workflow.downscaled.countrylevel`, and `workflow.history_aggregated.countrylevel`.

#### 11.3 Why Downscaling Depends on Harmonization

The `downscale()` function (in `concordia.downscale`) wraps the aneris `Downscaler`:

```python
downscaler = Downscaler(
    harmonized,          # ◄── harmonized regional trajectory (not raw model!)
    hist,                # country-level history
    base_year,
    regionmapping.data,  # country→region mapping
    luc_sectors=...,
    gdp=gdp,
)
methods = downscaler.methods()        # auto-select per sector
downscaled = downscaler.downscale(methods)
```

The `Downscaler` needs the harmonized data — not the raw model — because:

1. **`base_year_pattern`** (used for LUC/agriculture sectors): Computes weights as `w_c = hist_c(base_year) / Σ_c hist_c(base_year)` per region, then multiplies `result_c(t) = harmonized_region(t) × w_c`. If the regional total passed in were *not* harmonized, it would not match `Σ_c hist_c(base_year)` at `t = base_year`, breaking the fundamental identity that downscaled countries must sum to the regional total.

2. **`ipat_2100_gdp`** (used for most anthropogenic sectors): Emissions intensity `I_c = emissions_c / GDP_c` converges across countries. The starting intensity is anchored at the base year, so the regional trajectory must agree with observed history at that point.

In both cases, harmonization ensures that the regional-level trajectory is **anchored to the observed country-sum** in the base year, so the downscaling weights are consistent.

#### 11.4 The `method` Index Level

Both `harmonize()` and `downscale()` add a `method` level to the DataFrame index, recording which algorithm was used for each row:

- After harmonization: `method` = e.g., `"reduce_ratio_2080"`, `"constant_offset"`, etc.
- After downscaling: `method` = e.g., `"base_year_pattern"`, `"ipat_2100_gdp"`, etc.

The `method` level on `workflow.harmonized` is used by `harmonized_data.to_iamc()` to produce labelled columns in the export CSV. The `method` level on `workflow.downscaled` is dropped before gridding.

#### 11.5 Harmonization Algorithms

The `harmonize()` function (in `concordia.harmonize`) splits variables into LUC sectors and non-LUC sectors, then delegates to aneris `Harmonizer`:

| Algorithm | When used | Behaviour |
|-----------|-----------|-----------|
| `reduce_ratio_2080` | Non-LUC sectors (default) | The ratio `model / history` at the base year converges linearly to 1 by 2080. |
| `constant_ratio` | Override-selectable | The ratio `model / history` stays fixed at its base-year value for all future years. |
| `reduce_offset_2150_cov` | LUC sectors (default, with `default_luc_method` unset) | The absolute offset `model − history` converges to 0 by 2150, adjusted by coefficient of variation. |
| `constant_offset` | Override-selectable | The offset `model − history` stays fixed at its base-year value. |

> **Known issue:** Offset-based methods (`reduce_offset_2150_cov`, `constant_offset`) can produce **negative harmonized values** when the offset is larger than the model value in future years. This is the root cause of the negative-value problem observed in Agricultural Waste Burning for some regions. The interim fix clips negative values to zero after downscaling: `downscaled.clip(lower=0)`.

#### 11.6 Downscaling Algorithms

The `Downscaler` auto-selects methods based on sector classification:

| Algorithm | Sectors | Mechanism |
|-----------|---------|-----------|
| `base_year_pattern` | LUC sectors: Agricultural Waste Burning, Forest Burning, Grassland Burning, Peat Burning, Agriculture | `weight_c = hist_c(base_year) / Σ hist(base_year)` per region. Each country keeps its base-year share of the regional total for all years. |
| `ipat_2100_gdp` | All other anthropogenic sectors: Energy, Industrial, Transport, Residential, Solvents, Waste, etc. | Emissions intensity `I = emissions / GDP` converges across countries within each region by 2100. Countries with high intensity decrease; countries with low intensity increase. |

#### 11.7 The `harmonized_data` Property

After `harmonize_and_downscale()` completes, the `workflow.harmonized_data` property assembles a `Harmonized` object:

```python
@property
def harmonized_data(self):
    hist = self.history_aggregated.data          # concatenated history (all 3 levels)
    model = self.model.pix.semijoin(hist.index)  # raw model aligned to history index
    return Harmonized(
        hist=hist,
        model=model,
        harmonized=self.harmonized.data,         # concatenated harmonized (all 3 levels)
        skip_for_total=self.variabledefs.skip_for_total,
    )
```

This `Harmonized` object has methods:
- `.add_totals()` — compute sectoral and gas-level totals.
- `.to_iamc()` — format as IAMC-style DataFrame with `Unharmonized`, `Harmonized|{method}`, and historic rows side-by-side.

The export code calls:
```python
workflow.harmonized_data.add_totals().to_iamc(...)
```

This is why `workflow.harmonized` and `workflow.history_aggregated` **must** be populated — otherwise the export and the gridding step (which also reads `history_aggregated`) will fail.

#### 11.8 Skipping Harmonization (Pre-harmonized Input Data)

In the CMIP7 pipeline, the IAM input scenarios have **already been harmonized** upstream (by the IAM teams themselves). The `harm_overrides` file is empty and the concordia harmonization step is essentially a pass-through. However, aneris still applies default algorithms, which can introduce artefacts — notably **negative values** in sectors where offset-based methods push harmonized trajectories below zero.

The `patch_harmonization.py` module provides a `skip_harmonization` option that bypasses harmonization entirely:

```python
from patch_harmonization import patch_harmonize_and_downscale
patch_harmonize_and_downscale(workflow, skip_harmonization=True)
```

When `skip_harmonization=True`, the patch replaces `harmonize_and_downscale()` with a version that:

1. **Mirrors the 3-level structure** — calls `_skip_harmdown_globallevel()`, `_skip_harmdown_regionlevel()`, and `_skip_harmdown_countrylevel()`, which replicate the same filtering, aggregation, and missing-data handling as the originals.

2. **Uses raw model data instead of harmonized data** — instead of calling `concordia.harmonize.harmonize()`, the model data (trimmed to `base_year` onward) is used directly. A `method="skip"` level is added to the index so the downstream `to_iamc()` export still works.

3. **Populates all three `GlobalRegional` containers** — `workflow.harmonized`, `workflow.downscaled`, and `workflow.history_aggregated` are all set, so the `harmonized_data` property, CSV exports, quality checks, and gridding all work unchanged.

4. **Fills missing history with zeros** — if a country exists in the region mapping but has no historical emissions data, zero-filled rows are added. This prevents index-alignment crashes during downscaling (the `base_year_pattern` method would otherwise get NaN weights for missing countries). This is handled by `_fill_missing_history_with_zero()`.

5. **Calls `concordia.downscale.downscale()` normally** — the raw (unharmonized) model data is passed to the `Downscaler` in place of harmonized data. Because the input scenarios are already harmonized upstream, the base-year values should already match the country-sum of history, and downscaling weights remain valid.

The key data flow comparison:

```
NORMAL MODE (skip_harmonization=False):
  model ──► harmonize() ──► harm ──► downscale(harm, hist, ...) ──► downscaled
                                ▲
                                │
                           hist_agg (country→region aggregated)

SKIP MODE (skip_harmonization=True):
  model ──────────────────► model ──► downscale(model, hist, ...) ──► downscaled
        (trimmed to base_year+)  ▲
                                 │
                            hist (zero-filled for missing countries)
```

This mode is useful for:
- **Debugging:** Confirming whether negative values originate from harmonization or from the input data itself.
- **Production with pre-harmonized data:** When IAM teams have already harmonized their scenarios and no further adjustment is desired.

**Post-downscaling fix:** Regardless of mode, negative values are clipped to zero with `downscaled.clip(lower=0)`.

**Outputs saved:**
- `harmonization-{version}.csv` — harmonized (or skip-harmonized) regional data in IAMC format.
- `downscaled-only-{version}.csv` — country-level downscaled data.

### Step 12: Quality Control on Downscaled Data

Two checks are run on the downscaled data:

1. **No disallowed negatives:** All values must be ≥ 0, except for:
   - CDR sectors (`Direct Air Capture`, `Other CDR`, `Enhanced Weathering`, `BECCS`) — expected negative (removal).
   - `Industrial Sector` + `CO2` — can be negative (process emissions offsets).

2. **CDR sectors must be negative:** `Direct Air Capture`, `Enhanced Weathering`, and `BECCS` must have values ≤ 0 (they represent carbon removal).

A `ValueError` is raised if either check fails.

**Additional diagnostics:**
- Countries covered vs. missing.
- Share of global emissions missing due to un-downscaled territories.

### Step 13: Gridding to NetCDF

```python
res = workflow.grid(
    template_fn="...",
    callback=cmip7_utils.DressUp(...),
    directory=version_path,
    skip_exists=SKIP_EXISTING_MAIN_WORKFLOW_FILES,
)
```

For each output variable defined in `variabledefs`:
1. The relevant proxy raster (`IndexRaster`) distributes country-level emissions across grid cells.
2. Units are converted to `kg species s⁻¹ m⁻²`.
3. Results are saved as NetCDF with zlib compression.

**Performance:** ~1 hour for all 10 species for 1 scenario.
**Output size:** ~11.4 GB per scenario.

Grid cell areas are loaded from `areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc` and also exported with updated CMIP7 metadata.

### Step 14: Spatial Harmonization

**Goal:** Align the gridded 2023 spatial pattern with raw CEDS anthropogenic data, so country-border artifacts are minimized.

**Algorithm (per gas, per `em_anthro` file):**

1. **Ratio grid:** Compute the ratio `CEDS_2023 / gridded_2023` per grid cell per sector.
2. **Zero-fill:** Where gridded = 0 but CEDS ≠ 0 (e.g., overseas territories), add CEDS values directly.
3. **Apply ratio weights** to all years (expand 2023 ratio across the full time series).
4. **Global-total conservation:**
   - Compute a per-sector global scalar = `gridded_global / weighted_global`.
   - Linearly relax the correction from full effect in 2023 to zero by 2050, then no correction 2050–2100.
5. **Residual correction:** Any remaining difference between CEDS 2023 and the corrected field is also added and linearly relaxed to zero by 2050.
6. **2022 replacement:** The 2022 time slice is replaced entirely with raw CEDS 2022 data (since 2022 is a historical year).
7. Save back to the same NetCDF file (overwrite).

**Sectors excluded from ratio scaling:** `BECCS` and `Other Capture and Removal`.

### Step 15: Timeseries Corrections

Three independent corrections ensure gridded global totals match the input IAM totals exactly:

#### 15a. Aviation (AIR) Timeseries Correction

For each `*AIR*` NetCDF file:
1. Compute gridded annual global totals.
2. Compute input IAM annual totals (sum across regions).
3. Ratio = IAM total / gridded total, per sector per year.
4. Apply ratio as a uniform scalar across all grid cells.

#### 15b. Anthropogenic Timeseries Correction

Same approach as aviation, but for non-AIR, non-openburning, non-VOC files. Sector names are mapped from IAM conventions (`"Energy Sector"` → `"Energy"`, etc.). CDR sectors are aggregated into `"Other Capture and Removal"`.

Special handling: International Shipping ratios for 2022 and 2023 are forced to 1.0.

#### 15c. Openburning Timeseries Correction

Same approach, applied to `*openburning*` files (excluding speciated VOC and H₂).

### Step 16: H₂ Openburning Derivation

**Condition:** `run_openburning_h2 = True`

H₂ openburning emissions are not available directly from IAMs. Instead:

```
H2_openburning = CO_openburning × (EF_H2 / EF_CO)
```

where `EF_H2 / EF_CO` is a gridded emission-factor ratio file (`EF_h2_div_EF_co.nc`), varying by sector, year, and month.

The multiplication is performed sector-by-sector, time-step-by-time-step.

### Step 17: VOC Speciation

#### 17a. Openburning VOC Speciation

**Condition:** `run_openburning_supplemental_voc = True`

For each BB4CMIP7 VOC species:
```
speciated_VOC = NMVOCbulk_openburning × emissions_share(gas, sector, year, month)
```

Shares are loaded from `NMVOC_speciation/` proxy files. Output files follow the naming convention `NMVOC-{species}-em-speciated-VOC-openburning_{FILE_NAME_ENDING}`.

#### 17b. Anthropogenic VOC Speciation

**Condition:** `run_anthro_supplemental_voc = True`

Same approach but using CEDS VOC speciation shares from `VOC_speciation/` proxy files. Output files follow `{VOC_species}-em-speciated-VOC-anthro_{FILE_NAME_ENDING}`.

**Validation:** The sum of all speciated VOC species is checked against the bulk NMVOC total (tolerance: 0.1%).

### Step 18: Post-processing & Plotting

#### 18a. Annual Totals CSV Export

For each gridded NetCDF, compute global annual emissions totals (with and without sector breakdown) and save as CSV to `check_annual_totals/`.

#### 18b. Spatial Comparison Maps

For each gas × sector × year, generate 4-panel maps comparing:
1. CEDS historical data
2. CMIP7 scenario data
3. Difference (CEDS − Scenario)
4. Percentage difference

#### 18c. Timeseries Plots

For selected locations (e.g., Laxenburg, London, Lagos, South China Sea):
- Single grid-point timeseries comparing CEDS and scenario.
- Area-average timeseries (±2° lat/lon).

#### 18d. NMVOC Consistency Checks

- Verify speciated openburning NMVOC species sum to NMVOCbulk (per sector).
- Verify speciated anthro VOC species sum to NMVOC `em_anthro`.
- Compare downscaled totals against gridded totals via FacetGrid line plots.

---

## 4. Debug Patches

Two debug modules can be applied before `harmonize_and_downscale()`:

### `debug_base_year_pattern.py`

Monkey-patches `aneris.downscaling.methods.base_year_pattern` to:
- Log country coverage mismatches between history and region mapping.
- Log NaN / negative values in downscaling weights.
- Store all intermediate data (weights, model input, results) per call in `_DEBUG_INFO_LIST`.

**Usage:**
```python
from debug_base_year_pattern import patch_base_year_pattern, get_debug_info
patch_base_year_pattern(fill_missing_countries=False, log_diagnostics=True, return_debug_info=True)
```

**Options:**
- `fill_missing_countries=True`: Fill zero-valued rows for missing countries (fixes NaN weights).
- `log_diagnostics=True`: Print diagnostic info at 10+ checkpoints.
- `return_debug_info=True`: Store weights/results for later retrieval via `get_debug_info(sector="Agricultural Waste Burning")`.

### `patch_harmonization.py`

Monkey-patches `workflow.harmonize_and_downscale()` to either:

1. **Capture mode** (`skip_harmonization=False`): Intercept harmonized data before downscaling for inspection.
2. **Skip mode** (`skip_harmonization=True`): Bypass harmonization entirely — use raw model data, then downscale.

In skip mode, the patch:
- Replicates the 3-level structure (`globallevel`, `regionlevel`, `countrylevel`).
- Populates `workflow.harmonized` with raw model data + `method="skip"`.
- Populates `workflow.history_aggregated` with aggregated history.
- Populates `workflow.downscaled` with the downscaled result.
- Fills missing history with zeros so downscaling does not crash.

This ensures all downstream code (`workflow.harmonized_data`, CSV exports, quality checks) works unchanged.

---

## 5. Output Files

All outputs go under `{out_path}/{GRIDDING_VERSION}/`.

### Tabular Data

| File | Description |
|------|-------------|
| `scenarios_processed.csv` | Processed IAM data (wide format) |
| `harmonization-{version}.csv` | Harmonized regional data in IAMC format |
| `downscaled-only-{version}.csv` | Country-level downscaled data |
| `workflow_driver_data/` | Saved workflow inputs for reproducibility |

### Gridded NetCDF

| Pattern | Description |
|---------|-------------|
| `{GAS}-em-anthro_{FILE_NAME_ENDING}` | Anthropogenic emissions (10 species) |
| `{GAS}-em-openburning_{FILE_NAME_ENDING}` | Openburning emissions |
| `{GAS}-em-AIR-anthro_{FILE_NAME_ENDING}` | Aviation emissions |
| `H2-em-openburning_{FILE_NAME_ENDING}` | Derived H₂ openburning |
| `NMVOC-{species}-em-speciated-VOC-openburning_...` | Speciated openburning VOC |
| `{species}-em-speciated-VOC-anthro_...` | Speciated anthropogenic VOC |
| `areacella/areacella_...nc` | Grid cell area file |

### QC & Diagnostics

| Directory | Description |
|-----------|-------------|
| `check_annual_totals/` | Per-gas annual totals CSV |
| `check_annual_totals_ext/` | Extended totals comparing gridded vs downscaled |
| `check_NMVOC_sums/` | NMVOC speciation consistency checks |
| `check_VOC_sums/` | Anthro VOC speciation consistency checks |
| `plots/` | Timeseries and map comparison PNGs |
| `debug_*.csv` | Debug outputs from harmonization/downscaling patches |

---

## 6. Data Flow Diagram

```
                    ┌──────────────┐
                    │  YAML Config │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                  ▼
   ┌───────────┐    ┌────────────┐    ┌──────────────┐
   │ IAM Model │    │ Historical │    │ GDP (SSP DB) │
   │ Scenarios │    │ Emissions  │    │              │
   └─────┬─────┘    └─────┬──────┘    └──────┬───────┘
         │                │                   │
         │    ┌───────────┼───────────────────┘
         │    │           │
         ▼    ▼           ▼
     ┌────────────────────────────┐
     │      WorkflowDriver       │
     │  ┌──────────────────────┐ │
     │  │   Harmonization      │◄├──── harm_overrides (empty)
     │  │  (aneris methods)    │ │
     │  └──────────┬───────────┘ │
     │             ▼             │
     │  ┌──────────────────────┐ │
     │  │    Downscaling       │ │     Methods:
     │  │  base_year_pattern   │ │     - LUC/burning sectors
     │  │  ipat_2100_gdp       │ │     - All other sectors
     │  └──────────┬───────────┘ │
     │             ▼             │
     │  ┌──────────────────────┐ │
     │  │     Gridding         │ │     IndexRaster + proxy
     │  │  (ptolemy)           │ │     weights
     │  └──────────┬───────────┘ │
     └─────────────┼─────────────┘
                   ▼
          ┌────────────────┐
          │  NetCDF (raw)  │
          └────────┬───────┘
                   │
     ┌─────────────┼──────────────────┐
     ▼             ▼                  ▼
┌─────────┐  ┌──────────┐   ┌────────────────┐
│ Spatial  │  │Timeseries│   │ VOC Speciation │
│ Harmoni- │  │Correction│   │ + H₂ Deriv.    │
│ zation   │  │(anthro,  │   │                │
│(vs CEDS) │  │ AIR, OB) │   │                │
└────┬─────┘  └────┬─────┘   └───────┬────────┘
     │             │                  │
     └─────────────┼──────────────────┘
                   ▼
          ┌────────────────┐
          │ NetCDF (final) │
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │   QC + Plots   │
          └────────────────┘
```
