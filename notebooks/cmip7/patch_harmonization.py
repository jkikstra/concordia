"""
Patch to capture harmonized data before downscaling, or skip harmonization
entirely.

When skip_harmonization=True:
  - workflow.harmonized is populated with the raw model data (+ a 'method'
    level set to 'skip') so downstream code (harmonized_data property,
    export CSV) works unchanged.
  - workflow.history_aggregated is populated with the aggregated history.
  - workflow.downscaled is populated with the downscaled result.
  - Missing history rows are filled with zeros so downscaling never crashes
    on missing countries.
"""

import pandas as pd
from pandas_indexing import isin, concat as pix_concat

# Storage for harmonized data
_HARMONIZED_DATA = {}


def capture_harmonized_data(data, stage="before_downscaling"):
    """Store harmonized data for inspection."""
    global _HARMONIZED_DATA
    _HARMONIZED_DATA[stage] = data.copy() if hasattr(data, 'copy') else data
    print(f"[CAPTURE] Stored harmonized data at stage: {stage}")
    if isinstance(data, pd.DataFrame):
        print(f"  Shape: {data.shape}")
        if data.select_dtypes(include=['number']).shape[1] > 0:
            neg_count = (data.select_dtypes(include=['number']) < 0).sum().sum()
            if neg_count > 0:
                print(f"  [WARNING] Found {neg_count} negative values in harmonized data!")
    return data


def get_harmonized_data(stage="before_downscaling"):
    """Retrieve stored harmonized data."""
    return _HARMONIZED_DATA.get(stage, None)


# ------------------------------------------------------------------
# Helpers for the skip-harmonization path
# ------------------------------------------------------------------

def _fill_missing_history_with_zero(hist, reference_index, level="country"):
    """
    Ensure *hist* has rows for every label in *reference_index* at *level*.
    Missing rows are filled with 0.
    """
    # If hist is empty, nothing to fill from - return as-is
    if hist.empty:
        return hist

    existing = hist.index.get_level_values(level).unique()
    needed = reference_index.get_level_values(level).unique() if hasattr(
        reference_index, 'get_level_values') else pd.Index(reference_index)
    missing = needed.difference(existing)
    if missing.empty:
        return hist

    print(f"  [FILL] Adding zero-history for {len(missing)} missing {level}(s): "
          f"{list(missing[:10])}{'...' if len(missing) > 10 else ''}")

    # Build template from first row of hist, zeroed out
    template = hist.iloc[[0]].copy()
    template.iloc[:] = 0.0

    parts = [hist]
    # Other index levels to iterate over
    other_levels = [n for n in hist.index.names if n != level]
    if other_levels:
        other_combos = hist.index.droplevel(level).unique()
        for combo in other_combos:
            row = template.copy()
            if len(other_levels) == 1:
                combo = (combo,) if not isinstance(combo, tuple) else combo
            for lv_name, lv_val in zip(other_levels, combo):
                row = row.rename(index={row.index.get_level_values(lv_name)[0]: lv_val},
                                 level=lv_name)
            for m in missing:
                r = row.rename(index={row.index.get_level_values(level)[0]: m},
                               level=level)
                parts.append(r)
    else:
        for m in missing:
            r = template.rename(index={template.index.get_level_values(level)[0]: m},
                                level=level)
            parts.append(r)

    return pd.concat(parts).sort_index()


def _skip_harmdown_globallevel(workflow, variabledefs):
    """Replicate harmdown_globallevel but skip harmonization."""
    from concordia.utils import aggregate_subsectors

    variables = variabledefs.globallevel.index
    if variables.empty:
        return None

    print("[SKIP-GLOBAL] Processing global-level variables (no harmonization)")

    model = workflow.model.pix.semijoin(variables, how="right").loc[
        isin(region="World")
    ]
    hist = (
        workflow.hist.pix.semijoin(variables, how="right")
        .loc[isin(country="World")]
        .rename_axis(index={"country": "region"})
    )

    # Trim model to base_year onward (harmonize() does this)
    model = model.loc[:, workflow.settings.base_year:]

    # Use model as "harmonized", add method='skip'
    harmonized = model.pix.assign(method="skip")
    harmonized = aggregate_subsectors(harmonized)
    hist = aggregate_subsectors(hist)

    workflow.history_aggregated.globallevel = hist
    workflow.harmonized.globallevel = harmonized
    workflow.downscaled.globallevel = harmonized.pix.format(
        method="single", country="{region}"
    )

    return harmonized.droplevel("method").rename_axis(index={"region": "country"})


def _skip_harmdown_regionlevel(workflow, variabledefs):
    """Replicate harmdown_regionlevel but skip harmonization."""
    from concordia.utils import aggregate_subsectors

    vd = variabledefs.regionlevel
    if vd.empty:
        return None

    print("[SKIP-REGION] Processing region-level variables (no harmonization)")

    model = workflow.model.pix.semijoin(vd.index, how="right")
    hist = workflow.hist.pix.semijoin(vd.index, how="right")
    hist_agg = workflow.regionmapping.aggregate(hist, dropna=True)

    model = model.loc[:, workflow.settings.base_year:]
    model = model.loc[isin(region=workflow.regionmapping.data.unique())]

    harmonized = model.pix.assign(method="skip")
    harmonized = aggregate_subsectors(harmonized)
    hist_agg = aggregate_subsectors(hist_agg)

    workflow.history_aggregated.regionlevel = hist_agg
    workflow.harmonized.regionlevel = harmonized
    workflow.downscaled.regionlevel = harmonized.pix.format(
        method="single", country="{region}"
    )

    return harmonized.droplevel("method").rename_axis(index={"region": "country"})


def _skip_harmdown_countrylevel(workflow, variabledefs):
    """Replicate harmdown_countrylevel but skip harmonization."""
    from concordia.downscale import downscale
    from concordia.utils import aggregate_subsectors, add_zeros_like
    from pandas_indexing import concat

    history_aggregated = []
    harmonized_list = []
    downscaled_list = []

    print("[SKIP-COUNTRY] Processing country-level variables (no harmonization)")

    for group in workflow.country_groups(variabledefs):
        regionmapping = workflow.regionmapping.filter(group.countries)
        missing_regions = set(workflow.regionmapping.data.unique()).difference(
            regionmapping.data.unique()
        )
        missing_countries = workflow.regionmapping.data.index.difference(
            group.countries
        )

        model = workflow.model.pix.semijoin(group.variables, how="right")
        hist = workflow.hist.pix.semijoin(group.variables, how="right")

        # Fill missing history with zeros so downscaling has all countries
        countries_in_mapping = regionmapping.data.index
        hist = _fill_missing_history_with_zero(hist, countries_in_mapping, level="country")

        hist_agg = regionmapping.aggregate(hist, dropna=True)

        # Also fill aggregated hist for any regions that ended up empty
        regions_in_model = model.index.get_level_values("region").unique()
        hist_agg = _fill_missing_history_with_zero(hist_agg, regions_in_model, level="region")

        history_aggregated.append(
            add_zeros_like(hist_agg, hist, region=missing_regions)
        )

        # Model trimmed to base_year onward, filtered to mapped regions
        model_trimmed = model.loc[:, workflow.settings.base_year:]
        model_trimmed = model_trimmed.loc[isin(region=regionmapping.data.unique())]

        # Use model as "harmonized", add method='skip'
        harm = model_trimmed.pix.assign(method="skip")
        harmonized_list.append(
            add_zeros_like(harm, model, region=missing_regions, method=["all_zero"])
        )

        # Now aggregate subsectors and downscale
        harm_for_ds = aggregate_subsectors(model_trimmed)
        hist_for_ds = aggregate_subsectors(hist)

        down = downscale(
            harm_for_ds,
            hist_for_ds,
            workflow.gdp,
            regionmapping,
            settings=workflow.settings,
        )
        downscaled_list.append(
            add_zeros_like(
                down,
                harm_for_ds,
                country=missing_countries,
                method=["all_zero"],
                derive=dict(region=workflow.regionmapping.index),
            )
        )

    if not downscaled_list:
        return None

    workflow.history_aggregated.countrylevel = concat(history_aggregated)
    workflow.harmonized.countrylevel = concat(harmonized_list)
    ds = workflow.downscaled.countrylevel = concat(downscaled_list)

    return ds.droplevel(["method", "region"])


# ------------------------------------------------------------------
# Main patch function
# ------------------------------------------------------------------

def patch_harmonize_and_downscale(workflow, skip_harmonization=False):
    """
    Monkey-patch the harmonize_and_downscale method.

    Parameters
    ----------
    workflow : WorkflowDriver
        The concordia workflow object
    skip_harmonization : bool
        If True, skip harmonization entirely and only do downscaling.
        The model data is used as-is in place of harmonized data.
    """
    original_method = workflow.harmonize_and_downscale

    def wrapped_harmonize_and_downscale(*args, **kwargs):
        print("\n[PATCH] Intercepting harmonize_and_downscale...")

        if skip_harmonization:
            print("[SKIP] Harmonization disabled -- using original model data directly")
            from pandas_indexing import concat

            variabledefs = kwargs.get("variabledefs", None)
            if variabledefs is None and len(args) > 0:
                variabledefs = args[0]
            if variabledefs is None:
                variabledefs = workflow.variabledefs

            def skipnone(*dfs):
                return [d for d in dfs if d is not None]

            result = concat(
                skipnone(
                    _skip_harmdown_globallevel(workflow, variabledefs),
                    _skip_harmdown_regionlevel(workflow, variabledefs),
                    _skip_harmdown_countrylevel(workflow, variabledefs),
                )
            )

            print(f"[OK] Downscaling complete (harmonization was skipped)")
            print(f"  Result shape: {result.shape}")
            neg_count = (result < 0).sum().sum()
            if neg_count > 0:
                print(f"  [WARNING] Found {neg_count} negative values in downscaled result")
            else:
                print(f"  [OK] No negative values in result")

            return result

        # ----- Original behavior with data capture -----
        import aneris.downscaling.core as core

        original_downscale = core.Downscaler.downscale

        def patched_downscale(self, *ds_args, **ds_kwargs):
            """Intercept the downscale call to capture harmonized input."""
            if len(ds_args) > 0:
                model_data = ds_args[0]
                print(f"\n[INTERCEPT] Capturing model data going into downscaling...")
                if isinstance(model_data, pd.DataFrame):
                    print(f"  Shape: {model_data.shape}")
                    print(f"  Index names: {model_data.index.names}")
                    capture_harmonized_data(model_data, "harmonized_before_downscaling")

            return original_downscale(self, *ds_args, **ds_kwargs)

        core.Downscaler.downscale = patched_downscale

        try:
            result = original_method(*args, **kwargs)
            print("[PATCH] harmonize_and_downscale completed")
            return result
        finally:
            core.Downscaler.downscale = original_downscale

    workflow.harmonize_and_downscale = wrapped_harmonize_and_downscale
    status = ("WITH harmonization capture"
              if not skip_harmonization
              else "WITHOUT harmonization (skip mode)")
    print(f"[OK] Patched workflow.harmonize_and_downscale {status}")
