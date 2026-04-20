# Workflow Analysis: `workflow_cmip7-fast-track.py`

Analysis of the ~4200-line CMIP7 ScenarioMIP emissions workflow notebook.
Covers: (1) README outline for the workflow, (2) potential critical bugs, (3) improvement opportunities.

---

## 1. README Outline

### A. Overview
Single-scenario CMIP7 harmonization, downscaling, gridding, and post-processing pipeline for ScenarioMIP input4MIPs emissions. Run one scenario (marker) at a time. Driven either interactively or via a papermill/driver script.

### B. Configuration Parameters (lines 44–82)
- `HISTORY_FILE`: country-level historical emissions CSV
- `SETTINGS_FILE`: YAML config pointing to data paths (`config_cmip7_v0-4-0.yaml`)
- `VERSION_ESGF`, `GRIDDING_VERSION`: output version strings
- `marker_to_run`: scenario marker (`h`, `hl`, `m`, `ml`, `l`, `ln`, `vl`)
- Boolean flags: `run_main`, `run_main_gridding`, `run_spatial_harmonisation`, `run_anthro_timeseries_correction`, `run_AIR_anthro_timeseries_correction`, `run_openburning_timeseries_correction`, `run_openburning_h2`, `run_anthro_supplemental_voc`, `run_openburning_supplemental_voc`
- Species/sector filters: `DO_GRIDDING_ONLY_FOR_THESE_SPECIES`, `DO_GRIDDING_ONLY_FOR_THESE_SECTORS`

### C. Input Data
- Historical emissions: country-level CSV (`HISTORY_FILE`)
- IAM scenario: `harmonised-gridding_{MODEL}.csv`
- Region mappings: per-model CSV files
- Variable definitions: CSV/YAML of species+sector+proxy config
- GDP proxy: `ssp_basic_drivers_release_3.2.beta_full_gdp.csv` for downscaling
- Harmonization overrides: `harmonization_overrides.xlsx` (expected empty)
- Grid proxies: index raster, areacella, spatial proxy NetCDFs
- CEDS/BB4CMIP reference: raw NetCDF files for spatial harmonization and QC

### D. Workflow Steps (in execution order)
1. **Settings & logging** (lines 173–265): load YAML config, create output directory, set up file+stream logging
2. **Variable definitions** (lines 279–329): load and optionally filter species/sectors
3. **Region mappings** (lines 336–352): load per-model country→region mappings; deduplicate
4. **Historical data** (lines 360–395): read CSV, split non-global vs. global (shipping/aircraft), compute World total; apply `country_combinations`
5. **IAM data** (lines 406–460): read pre-harmonized scenario CSV; filter to one scenario; parse `Emissions|{gas}|{sector}` variable names
6. **Harmonization overrides** (lines 472–479): load override file; assert it is empty (data already harmonized upstream)
7. **GDP proxy** (lines 507–674): read SSP GDP data; convert country names to ISO3; interpolate to annual resolution; extend with historical 2020 values; align with scenario via SSP-marker mapping
8. **Country/sector coverage checks** (lines 680–745): identify countries with matching GDP+history+regionmapping; assert CDR sectors are the only missing ones; add zero CDR history
9. **Dask setup** (lines 755–760): conditional threaded/distributed client
10. **Workflow object** (lines 768–922): instantiate `WorkflowDriver`; save info; check region coverage; check harmonization consistency (6× with different filters)
11. **Harmonize & downscale** (lines 1157–1504): call `workflow.harmonize_and_downscale()`; apply three post-downscaling fixes (small AWB negatives, near-zero global shipping/aircraft, small non-CO2 negatives); monkey-patch `harmonize_and_downscale` to return fixed result; export harmonized and downscaled CSVs; run QC checks (negative/positive value assertions, near-zero global totals)
12. **Main gridding** (lines 1515–1530): call `workflow.grid()` to produce NetCDF files per species/sector
13. **areacella** (lines 1544–1573): update metadata and save area-of-gridcell NetCDF
14. **Spatial harmonization** (lines 1688–1963, `run_spatial_harmonisation`): for each anthro NetCDF file, compute ratio vs. CEDS 2023 reference; apply multiplicative correction + additive (overseas territories) correction; linearly taper correction to zero by 2050; replace 2022 slice with raw CEDS; overwrite file
15. **AIR timeseries correction** (lines 1978–2135, `run_AIR_anthro_timeseries_correction`): compute ratio of IAM aircraft totals to gridded aircraft totals; apply as global scalar per year; overwrite file
16. **Anthro timeseries correction** (lines 2141–2323, `run_anthro_timeseries_correction`): same logic for anthropogenic sectors; handles N2O unit conversion; locks sector 7 (shipping) ratio to 1.0 for 2022–2023; overwrite file
17. **Openburning timeseries correction** (lines 2329–2487, `run_openburning_timeseries_correction`): same logic for openburning sectors; overwrite file
18. **H2 openburning** (lines 2575–2684, `run_openburning_h2`): multiply CO openburning grid by H2/CO emission factor ratios (time-step loop); save H2 NetCDF
19. **VOC speciation — openburning** (lines 2812–2941, `run_openburning_supplemental_voc`): vectorized multiplication of NMVOCbulk openburning by per-species shares (BB4CMIP format); save per-species NetCDFs
20. **VOC speciation — anthro** (lines 2981–3102, `run_anthro_supplemental_voc`): same for CEDS anthropogenic NMVOC; save per-species NetCDFs
21. **Post-processing plots — maps** (lines 3168–3485): 4-column comparison maps (CEDS vs. scenario, difference, % difference) per gas/sector for January 2023
22. **Post-processing — annual totals CSV** (lines 3498–3536): global annual total emissions per sector for each NetCDF
23. **Post-processing — gridded vs. IAM comparison** (lines 3643–3886): combine all gridded totals; align with IAM totals; compute and save difference/relative-difference CSVs; facet line plots per gas/sector
24. **Post-processing — NMVOC speciation sum check** (lines 3892–4039): verify speciated BB4CMIP VOC sums to NMVOCbulk; plot vs. downscaled
25. **Post-processing — anthro VOC speciation sum check** (lines 4045–4210): verify speciated CEDS VOC sums to NMVOC_em_anthro; plot vs. downscaled

### E. Output Files
- `harmonization-{version}.csv`: harmonized IAMC-format data
- `downscaled-only-{version}.csv`: country-level downscaled data
- `{gas}-em-{sector}_{FILE_NAME_ENDING}.nc`: gridded NetCDF files (main + H2 + VOC speciation)
- `areacella_*.nc`: grid cell area file with updated metadata
- `plots/`: PNG maps and timeseries comparison plots
- `check_annual_totals/`, `check_annual_totals_ext/`, `check_NMVOC_sums/`, `check_VOC_sums/`: QC CSV files and plots
- Log: `debug_{version}.log`

---

## 2. Potential Critical Bugs

<!-- ### B1. Accessing closed dataset `gridded` after closing it (line 1939)
**Location:** [workflow_cmip7-fast-track.py:1939](workflow_cmip7-fast-track.py#L1939)
```python
# Earlier (lines 1906-1908):
if 'gridded' in locals():
    gridded.close()
# ...
# Line 1939:
emissions_harmonised[var].attrs = gridded[var].attrs.copy()
```
After explicitly closing `gridded`, the code accesses `gridded[var].attrs`. While xarray may retain in-memory attrs for already-loaded data, this is fragile and could fail if the data was lazy/dask-backed and not yet materialized.

### B2. Opening H2 CO dataset then immediately closing it (lines 2580–2581)
**Location:** [workflow_cmip7-fast-track.py:2580](workflow_cmip7-fast-track.py#L2580)
```python
co_openburning = xr.open_dataset(co_openburning_file)
co_openburning.close()  # ← closed immediately
```
Then `co_openburning` is used throughout the H2 openburning loop (lines 2589, 2602–2636). If any of the data is lazy (dask-backed), it would fail at access time with a closed-file error. The `.close()` call should be at the end of the block. -->

### B3. Triple overwrite of `CALCULATE_TOTALS_GASES` making it a string (lines 3500–3502)
**Location:** [workflow_cmip7-fast-track.py:3500](workflow_cmip7-fast-track.py#L3500)
```python
CALCULATE_TOTALS_GASES: list[str] | None = None
CALCULATE_TOTALS_GASES: list[str] | None = list(dict.fromkeys(GASES_ESGF_CEDS + GASES_ESGF_BB4CMIP))
CALCULATE_TOTALS_GASES: list[str] | None = "NMVOCbulk"   # ← wins, is a string!
```
At line 3517: `if gas_name in CALCULATE_TOTALS_GASES:` — because `CALCULATE_TOTALS_GASES` is the string `"NMVOCbulk"`, `in` does substring matching. `gas_name = "NMVOC"` would match (`"NMVOC" in "NMVOCbulk"` is `True`), silently including the wrong gas.

### B4. `assert remainder_diff_2023 < 50` doesn't check the negative side (line 1863)
**Location:** [workflow_cmip7-fast-track.py:1863](workflow_cmip7-fast-track.py#L1863)
```python
assert remainder_diff_2023 < 50  # Mt / year
```
`remainder_diff_2023 = ref2023 - gridded2023`. If the scenario is much larger than the CEDS reference (negative remainder), the assertion passes even though the discrepancy is large. Should be `assert abs(remainder_diff_2023) < 50`.

### B5. `experiment_name` may be undefined when `run_main_gridding=False` but `run_openburning_h2=True` (line 2646)
**Location:** [workflow_cmip7-fast-track.py:1517](workflow_cmip7-fast-track.py#L1517) (set) vs [line 2646](workflow_cmip7-fast-track.py#L2646) (used)

`experiment_name` is assigned at line 1517 inside `if run_main_gridding:`. If `run_main_gridding = False` and `run_openburning_h2 = True`, line 2646 raises `NameError: name 'experiment_name' is not defined`.

### B6. Duplicate identical call to `check_harmonization_consistency` (lines 1104 and 1107)
**Location:** [workflow_cmip7-fast-track.py:1104](workflow_cmip7-fast-track.py#L1104)
```python
check_harmonization_consistency(workflow, settings, version_path)  # line 1104
# Check all regions (original behavior)
check_harmonization_consistency(workflow, settings, version_path)  # line 1107 (identical)
```
The function is called twice with the same args, doubling runtime and overwriting the same output files.

### B7. `new_stem` computed from potentially unset or wrong `file` loop variable (lines 3738, 3950)
**Location:** [workflow_cmip7-fast-track.py:3738](workflow_cmip7-fast-track.py#L3738)
```python
parts = file.stem.split("_")
new_stem = "_".join(parts[1:])
```
`file` is the loop variable from the preceding `for file in tqdm(...)` loop. If the loop ran zero iterations, this raises `NameError`. If the variable carries over from a prior loop, it silently uses the wrong file.

### B8. `_what_emissions_variable_type` may return unbound `type` (lines 1646–1651)
**Location:** [workflow_cmip7-fast-track.py:1646](workflow_cmip7-fast-track.py#L1646)
```python
def _what_emissions_variable_type(file, files_main=[], files_voc=[]):
    if file in files_main:
        type = "em_anthro"
    elif file in files_voc:
        type = "em_speciated_VOC_anthro"
    return type  # ← UnboundLocalError if file is in neither list
```
If `file` is in neither list, `type` is never assigned and `return type` raises `UnboundLocalError`.

---

## 3. Things That Can Be Improved

### Code quality
- **I1.** Multiple blocks of 28-line `#-----` dividers (around lines 2494, 2688, 2762, 2944, 3108, etc.) add visual noise but carry no information. Replace with a single blank line or a clear section header.
- **I2.** `GRIDDING_VERSION` is assigned twice on adjacent lines (55–56): first `None`, then immediately overwritten. Remove the first assignment.
- **I3.** Large commented-out code blocks should be removed: old CMIP6 SSP GDP code (lines 607–627), old `select_only_countries_with_all_info` function (lines 703–714), manual `harmdown_*` steps (lines 1139–1149).
- **I4.** The `merged` DataFrame (GDP with historical 2020 appended, lines 580–591) is computed but never used — `gdp` is overwritten at line 600 via a different code path. Either use `merged` for interpolation, or remove the merge block.
- **I5.** `CALCULATE_TOTALS_GASES` is set three times on consecutive lines (3500–3502), clearly leftover from iterative development. Should be a single, unambiguous assignment.
- **I6.** The `rename_gdp` dict (lines 639–657) works around `pycountry` not recognizing certain country names. The code already notes `nomenclature-iamc` as the proper fix (line 636) — this should be resolved properly.

### Performance
- **I7.** The H2 openburning computation (lines 2619–2636) uses a Python loop over every time step (`for time_idx, time_val in enumerate(...)`). For 228+ monthly timesteps × 4 sectors × lat × lon, this is very slow. The xarray multiplication `co_slice * translation_slice` is already vectorizable; the loop can be replaced with a single aligned multiply.
- **I8.** `ds_to_annual_emissions_total` (described as taking 10–30 seconds) is called 5× inside the spatial harmonization loop (once per file). Lazy computation paths or caching could reduce this bottleneck.
- **I9.** Three separate timeseries correction blocks (AIR, anthro, openburning, lines 1978–2487) share nearly identical structure. They could be unified into a single `apply_timeseries_correction(files, sector_dict, gas_filter, ...)` helper function.

### Robustness
- **I10.** The `regionmapping` variable used at line 686 is the last value from the `for m, kwargs in settings.regionmappings.items()` loop (line 338). This is fragile — if multiple models are ever supported, it silently uses the wrong mapping.
- **I11.** Datasets opened inside loops (`xr.open_dataset`) are frequently not managed with `with` statements. Resource leaks are likely if an exception is raised mid-loop.
- **I12.** `run_main` does not guard the harmonization consistency checks (lines 1103–1125, which always run). If `run_main=False`, `workflow.model` may not be properly initialized, causing those checks to fail.
- **I13.** The monkey-patch of `workflow.harmonize_and_downscale` at line 1280 is fragile. If `workflow.grid()` calls the method through an internal reference (e.g., `self.harmonize_and_downscale`), the external patch won't apply and the fixes will be bypassed.

### Documentation / structure
- **I14.** At ~4200 lines, the notebook is very long. Natural split points exist: (a) data loading & preparation, (b) harmonization+downscaling, (c) gridding, (d) post-processing corrections, (e) QC/plotting. These could be separate notebooks driven by a thin orchestration script.
- **I15.** `check_harmonization_consistency` (lines 943–1101, ~160 lines) is defined inline in the notebook. It could be moved to `concordia/cmip7/utils.py` to keep the notebook focused.
