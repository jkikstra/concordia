# Workflow Analysis: Update

Re-checks every point from `WORKFLOW_ANALYSIS.md` against the current
`workflow_cmip7-fast-track.py` (4198 lines). Line numbers below refer to the *current* file,
not the original doc's numbering — the file has changed shape since.

B1/B2 are omitted here: they were already struck through (HTML-commented) in the original doc.

---

## 1. Bugs

### B3. `CALCULATE_TOTALS_GASES` triple-overwrite — **RESOLVED**
Lines 3488–3489 now only assign it twice: `None`, then the real gas list
(`list(dict.fromkeys(GASES_ESGF_CEDS + GASES_ESGF_BB4CMIP))`). The third assignment that made it
the string `"NMVOCbulk"` is gone. The `in` check at line 3502 now does correct list membership.

### B4. `assert remainder_diff_2023 < 50` ignores the negative side — **PERSISTS**
Still exactly as described, at line 1848: `assert remainder_diff_2023 < 50 # Mt / year`. Still no
`abs()`, so a scenario much larger than the CEDS reference still passes silently.

### B5. `experiment_name` undefined if `run_main_gridding=False` but `run_openburning_h2=True` — **PERSISTS**
`experiment_name` is still only assigned inside `if run_main_gridding:` (line 1502), and still
used unconditionally inside the H2-openburning block, e.g. line 2633
(`h2_openburning.attrs['title'] = f"...{experiment_name}"`), which is gated only by
`run_openburning_h2`. Same `NameError` risk as before.

### B6. Duplicate identical `check_harmonization_consistency` call — **PERSISTS**
Lines 1089 and 1092 are still back-to-back identical calls:
`check_harmonization_consistency(workflow, settings, version_path)`, with the same
"Check all regions (original behavior)" comment in between. Still doubles that runtime and
overwrites the same output file twice.

### B7. `new_stem` from a possibly stale/unset loop variable — **PERSISTS**
The pattern (`parts = file.stem.split("_")` / `new_stem = "_".join(parts[1:])` right after a
`for file in tqdm(...)` loop ends, at the same indent level as the loop body rather than inside
it) still appears three times now — lines 3724, 3936, 4083 — up from two in the original review.
Same `NameError`-on-zero-iterations / wrong-file-on-carryover risk, now in one more place.

### B8. `_what_emissions_variable_type` can return unbound `type` — **PERSISTS**
Line 1631–1636, unchanged: no `else` branch, so `return type` raises `UnboundLocalError` if
`file` is in neither `files_main` nor `files_voc`.

---

## 2. Improvements

### Code quality

- **I1.** Divider-comment blocks — **PERSISTS**. ~140 lines in the file still match the
  `#----`/`# ----` pattern; not cleaned up.
- **I2.** `GRIDDING_VERSION` assigned twice adjacently — **PERSISTS** (now lines 40–41):
  `None`, then immediately overwritten. Same pattern, different line numbers.
- **I3.** Large commented-out blocks — **PERSISTS**. All three still present: the old CMIP6 GDP
  code (~line 594), the old `select_only_countries_with_all_info` function (commented at line
  688, and still referenced in comments at 892/894), and the manual `harmdown_*` steps
  (~lines 1127–1133).
- **I4.** `merged` DataFrame computed but unused — **PERSISTS**. Still built at lines 566–575
  (GDP + historical 2020, reindexed to year columns), but `gdp` is derived independently via
  `gdp.interpolate(...)` right after — `merged` is never referenced again except in a commented-out
  `# merged` inspection line.
- **I5.** Same issue as B3 — **RESOLVED** (see B3 above).
- **I6.** `rename_gdp` pycountry workaround — **PERSISTS**. Still present at line 624, same
  comment about the `nomenclature-iamc` package being the "proper" future fix.

### Performance

- **I7.** H2-openburning per-timestep Python loop — **PERSISTS**. The
  `for time_idx, time_val in enumerate(co_sector.time.values):` loop is still there
  (~line 2605), still not vectorized.
- **I8.** `ds_to_annual_emissions_total` called repeatedly without caching — **PERSISTS**. Now
  called 14 times across the file (up from the 5 originally flagged in the spatial-harmonization
  loop alone); no memoization added.
- **I9.** Three near-identical timeseries-correction blocks — **PERSISTS**. `run_AIR_anthro_timeseries_correction`
  (line 1965), `run_anthro_timeseries_correction` (line 2128), and
  `run_openburning_timeseries_correction` (line 2316) remain three separate inline blocks; no
  shared `apply_timeseries_correction(...)` helper was introduced.

### Robustness

- **I10.** `regionmapping` fragile last-loop-value — **RESOLVED**. The loop at line 320 now
  builds a proper `regionmappings = {}` dict keyed by model name, and the mapping actually used
  later is looked up correctly via `regionmapping = regionmappings[model_name]` (line 754) against
  the scenario's real model. A new comment now explicitly documents the (unrelated,
  intentional) constraint that only one model can be run at a time.
- **I11.** Datasets opened without `with` — **PERSISTS**. 29 `xr.open_dataset(...)` calls in the
  file, 0 using a `with` block.
- **I12.** Harmonization consistency checks not guarded by `run_main` — **PERSISTS**. There is
  now an `if run_main:` at line 914, but it only wraps `workflow.save_info(...)` — the
  region-mapping assertion and all `check_harmonization_consistency(...)` calls (lines 920–1107)
  remain at top-level indentation, unguarded.
- **I13.** Monkey-patch of `workflow.harmonize_and_downscale` — **PERSISTS** (pattern unchanged).
  Still `workflow.harmonize_and_downscale = lambda variabledefs=None: _fixed_downscaled`
  (line 1265). Same caveat as before: whether this is actually fragile depends on how
  `WorkflowDriver.grid()` internally invokes `self.harmonize_and_downscale` in
  `src/concordia/workflow.py`, which wasn't re-checked here.

### Documentation / structure

- **I14.** ~4200-line single notebook — **PERSISTS**. File is 4198 lines, still one script; no
  split into separate stages.
- **I15.** `check_harmonization_consistency` defined inline — **PERSISTS**. Still defined in the
  notebook itself (line 928), not moved to `concordia/cmip7/utils.py`.

---

## Summary

| | Resolved | Persists |
|---|---|---|
| Bugs (B3–B8) | 1 (B3) | 5 (B4, B5, B6, B7, B8) |
| Improvements (I1–I15) | 1 (I10) | 13 |

Only two of the original 20 tracked points have actually been fixed: the `CALCULATE_TOTALS_GASES`
string bug (B3/I5) and the fragile `regionmapping` variable (I10). Everything else — including
the two duplicate/undefined-variable bugs (B5, B6) and the unbound-variable risks (B7, B8) — is
still present in the current 4198-line file.
