# -*- coding: utf-8 -*-
"""
Debug replacement for aneris.downscaling.methods.base_year_pattern

This module provides a patched version of base_year_pattern that:
1. Logs diagnostic information about country coverage mismatches
2. Logs diagnostic information about NaN/negative weights
3. Optionally fills missing countries with zero to prevent issues

Usage:
------
Import this at the top of your workflow script AFTER importing aneris:

    from debug_base_year_pattern import patch_base_year_pattern
    patch_base_year_pattern(fill_missing_countries=True)  # or False for diagnostics only

Then run your workflow as normal.
"""

import logging
from typing import Union
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Also print to console for immediate visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# Global storage for debug info - MUST be defined before functions that use it
# Changed to list to store all calls, not just the last one
_DEBUG_INFO_LIST = []
_CALL_COUNT = 0


def normalize(s: Series) -> Series:
    """Normalize a series to sum to 1, handling zero sums."""
    total = s.sum()
    if total == 0:
        return s * 0  # Return zeros if sum is zero
    return s / total


def base_year_pattern_debug(
    model: DataFrame,
    hist: Union[Series, DataFrame],
    context,  # DownscalingContext
    fill_missing_countries: bool = False,
    log_diagnostics: bool = True,
    return_debug_info: bool = False,
) -> DataFrame:
    """
    Debug version of base_year_pattern that diagnoses country coverage issues.

    Parameters
    ----------
    model : DataFrame
        Model emissions for each world region and trajectory
    hist : DataFrame or Series
        Historic emissions for each country and trajectory
    context : DownscalingContext
        Settings for downscaling, like the regionmap
    fill_missing_countries : bool, default False
        If True, fill NaN values (from missing countries) with 0 before normalizing.
        If False, keep original behavior but log warnings.
    log_diagnostics : bool, default True
        If True, log detailed diagnostics about coverage issues.
    return_debug_info : bool, default False
        If True, store debug info in global _DEBUG_INFO dict.

    Returns
    -------
    DataFrame:
        Downscaled emissions for countries
    """
    global _DEBUG_INFO, _CALL_COUNT
    from pandas_indexing import semijoin

    _CALL_COUNT += 1
    print(f"\n[DEBUG] base_year_pattern_debug CALLED (call #{_CALL_COUNT})")
    print(f"   return_debug_info={return_debug_info}")
    print(f"   model shape: {model.shape}")
    print(f"   hist type: {type(hist)}, shape: {hist.shape if hasattr(hist, 'shape') else 'N/A'}")
    print(f"   context.year: {context.year}")

    model_full = model.copy()  # Keep full model for diagnostics
    
    # CHECK MODEL INPUT FOR NEGATIVES
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("CHECKING MODEL INPUT VALUES (before any operations)")
        print(f"{'='*70}")
        neg_in_model = (model_full < 0).any(axis=1)
        if neg_in_model.any():
            print(f"  [WARNING] Found {neg_in_model.sum()} rows with negatives in MODEL INPUT!")
            # Show which years have negatives
            for idx in model_full[neg_in_model].index[:3]:  # Show first 3
                row = model_full.loc[idx]
                neg_years = row[row < 0]
                print(f"  Row: {idx}")
                print(f"    Negative years: {len(neg_years)}")
                print(f"    Sample: {dict(list(neg_years.items())[:5])}")
        else:
            # Check for exact zeros vs very small values
            zero_rows = (model_full == 0).any(axis=1)
            tiny_rows = ((model_full != 0) & (model_full.abs() < 1e-10)).any(axis=1)
            print(f"  [OK] No negative values in model input")
            print(f"  Rows with exact zeros: {zero_rows.sum()}")
            print(f"  Rows with tiny values (<1e-10): {tiny_rows.sum()}")
        print(f"{'='*70}\n")
    
    model = model.loc[:, context.year:]
    if isinstance(hist, DataFrame):
        hist_base_year = hist.loc[:, context.year]
    else:
        hist_base_year = hist

    # =========================================================================
    # DIAGNOSTIC 1: Check country coverage BEFORE semijoin
    # =========================================================================
    if log_diagnostics:
        # Get countries from each source
        hist_countries = set(hist_base_year.index.get_level_values("country").unique())
        # context.regionmap is already a MultiIndex, not a DataFrame
        regionmap_countries = set(context.regionmap.get_level_values("country").unique())
        
        # Find mismatches
        in_hist_not_regionmap = hist_countries - regionmap_countries
        in_regionmap_not_hist = regionmap_countries - hist_countries
        
        if in_hist_not_regionmap or in_regionmap_not_hist:
            print(f"\n{'='*70}")
            print("DIAGNOSTIC: Country coverage mismatch detected!")
            print(f"{'='*70}")
            
            if in_hist_not_regionmap:
                print(f"\n⚠️  Countries in HIST but NOT in REGIONMAP ({len(in_hist_not_regionmap)}):")
                print(f"    {sorted(in_hist_not_regionmap)}")
                print("    → These countries will be DROPPED from downscaling")
            
            if in_regionmap_not_hist:
                print(f"\n⚠️  Countries in REGIONMAP but NOT in HIST ({len(in_regionmap_not_hist)}):")
                print(f"    {sorted(in_regionmap_not_hist)}")
                print("    → These countries will get NaN weights after semijoin")
                print("    → This can cause negative results if NaN handling is problematic!")
            
            print(f"\n    Countries in both: {len(hist_countries & regionmap_countries)}")
            print(f"{'='*70}\n")

    # =========================================================================
    # Perform semijoin (this is where country mismatches become NaN)
    # =========================================================================
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("BEFORE SEMIJOIN:")
        print(f"{'='*70}")
        print(f"  hist_base_year shape: {hist_base_year.shape}")
        print(f"  hist_base_year index levels: {hist_base_year.index.names}")
        print(f"  context.regionmap shape: {len(context.regionmap)}")
        print(f"  context.regionmap levels: {context.regionmap.names}")
        
        # Check specifically for China/Taiwan countries
        china_taiwan_countries = ['chn', 'hkg', 'mac', 'twn']
        for country in china_taiwan_countries:
            country_in_hist = country in hist_base_year.index.get_level_values('country')
            country_in_regionmap = country in context.regionmap.get_level_values('country')
            print(f"  {country}: in hist={country_in_hist}, in regionmap={country_in_regionmap}")
        print(f"{'='*70}\n")
    
    hist_aligned = semijoin(hist_base_year, context.regionmap, how="right")
    
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("AFTER SEMIJOIN:")
        print(f"{'='*70}")
        print(f"  hist_aligned shape: {hist_aligned.shape}")
        print(f"  hist_aligned index levels: {hist_aligned.index.names}")
        
        # Check specifically for China/Taiwan countries AFTER semijoin
        china_taiwan_countries = ['chn', 'hkg', 'mac', 'twn']
        for country in china_taiwan_countries:
            country_mask = hist_aligned.index.get_level_values('country') == country
            count = country_mask.sum()
            print(f"  {country}: {count} rows")
            if count > 0 and count < 10:
                # Show a few examples
                sample = hist_aligned[country_mask].head(3)
                print(f"    Sample indices: {list(sample.index[:3])}")
                print(f"    Sample values: {list(sample.values[:3])}")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # DIAGNOSTIC 1.5: Check for ZERO historical values (countries present but zero emissions)
    # =========================================================================
    if log_diagnostics:
        # Check which countries have zero emissions in hist_aligned
        zero_mask = (hist_aligned == 0)
        zero_count = zero_mask.sum()
        
        if zero_count > 0:
            print(f"\n{'='*70}")
            print(f"DIAGNOSTIC: Found {zero_count} ZERO values in historical data after semijoin!")
            print(f"{'='*70}")
            
            # Show which trajectories have zeros
            zero_rows = hist_aligned[zero_mask]
            zero_summary = zero_rows.reset_index()
            
            # Get unique combinations of gas+sector with zeros
            if 'gas' in zero_summary.columns and 'sector' in zero_summary.columns:
                zero_gas_sector = zero_summary[['gas', 'sector']].drop_duplicates()
                print(f"\n  Gas+Sector combinations with zeros:")
                for _, row in zero_gas_sector.head(10).iterrows():
                    gas, sector = row['gas'], row['sector']
                    # Count how many countries have zero for this gas+sector
                    mask = (zero_summary['gas'] == gas) & (zero_summary['sector'] == sector)
                    n_countries = len(zero_summary[mask])
                    countries = zero_summary[mask]['country'].unique()[:5]  # Show first 5
                    print(f"    {gas}/{sector}: {n_countries} countries with zero (e.g., {list(countries)})")
            
            print(f"{'='*70}\n")
    
    # =========================================================================
    # DIAGNOSTIC 2: Check for NaN values AFTER semijoin (missing countries)
    # =========================================================================
    if log_diagnostics:
        nan_mask = hist_aligned.isna()
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            print(f"\n{'='*70}")
            print(f"DIAGNOSTIC: Found {nan_count} NaN values after semijoin!")
            print(f"{'='*70}")
            
            # Show which trajectories have NaN
            nan_rows = hist_aligned[nan_mask]
            nan_summary = nan_rows.reset_index()
            
            # Get unique values for each index level
            for col in nan_summary.columns:
                if col in hist_aligned.index.names:
                    unique_vals = nan_summary[col].unique()
                    if len(unique_vals) <= 20:
                        print(f"  {col}: {sorted(unique_vals)}")
                    else:
                        print(f"  {col}: {len(unique_vals)} unique values")
            
            print(f"\nFirst 10 NaN entries:\n{nan_rows.head(10)}")
            print(f"{'='*70}\n")

    # =========================================================================
    # DIAGNOSTIC 3: Check for negative historic values
    # =========================================================================
    if log_diagnostics:
        negative_hist = hist_aligned[hist_aligned < 0]
        if len(negative_hist) > 0:
            print(f"\n{'='*70}")
            print(f"DIAGNOSTIC: Found {len(negative_hist)} negative historic values!")
            print(f"{'='*70}")
            print(f"\nNegative values:\n{negative_hist.head(20)}")
            if len(negative_hist) > 20:
                print(f"  ... and {len(negative_hist) - 20} more")
            print(f"{'='*70}\n")

    # =========================================================================
    # Calculate weights with optional NaN filling
    # =========================================================================
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("BEFORE GROUPBY/NORMALIZE (calculating weights):")
        print(f"{'='*70}")
        print(f"  model.index.names (for groupby): {model.index.names}")
        print(f"  Number of unique groups: {hist_aligned.groupby(model.index.names, dropna=False).ngroups}")
        
        # Check a specific group - find China/Taiwan region with Peat or AWB
        for idx in model.index[:10]:
            if any('China' in str(x) or 'Taiwan' in str(x) for x in idx):
                if any('Peat' in str(x) or 'Agricultural Waste' in str(x) for x in idx):
                    print(f"\n  Example group index: {idx}")
                    # Get all hist_aligned rows for this group
                    group_mask = True
                    for i, name in enumerate(model.index.names):
                        if name in hist_aligned.index.names:
                            group_mask = group_mask & (hist_aligned.index.get_level_values(name) == idx[i])
                    
                    group_data = hist_aligned[group_mask]
                    print(f"    Rows in this group: {len(group_data)}")
                    print(f"    Countries: {list(group_data.index.get_level_values('country').unique())}")
                    print(f"    Values: {list(group_data.values)}")
                    print(f"    Sum: {group_data.sum()}")
                    break
        print(f"{'='*70}\n")
    
    if fill_missing_countries:
        # Fill NaN with 0 before normalizing - missing countries get zero weight
        hist_filled = hist_aligned.fillna(0)
        
        weights = (
            hist_filled
            .groupby(model.index.names, dropna=False)
            .transform(normalize)
        )
        
        if log_diagnostics and hist_aligned.isna().any():
            print("Applied fill_missing_countries=True: NaN values filled with 0 before normalization.")
    else:
        # Original behavior - NaN propagates through normalization
        weights = (
            hist_aligned
            .groupby(model.index.names, dropna=False)
            .transform(normalize)
        )
    
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("AFTER GROUPBY/NORMALIZE (weights calculated):")
        print(f"{'='*70}")
        print(f"  weights shape: {weights.shape}")
        
        # Check specifically for China/Taiwan countries in weights
        china_taiwan_countries = ['chn', 'hkg', 'mac', 'twn']
        for country in china_taiwan_countries:
            country_mask = weights.index.get_level_values('country') == country
            count = country_mask.sum()
            print(f"  {country}: {count} weight rows")
            if count > 0:
                country_weights = weights[country_mask]
                non_zero = (country_weights != 0).sum()
                print(f"    Non-zero weights: {non_zero}")
                if count < 10:
                    print(f"    Sample weights: {list(country_weights.values[:5])}")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # DIAGNOSTIC 3.5: Special check for China and Taiwan region with Peat Burning
    # =========================================================================
    if log_diagnostics:
        # Check if we have any China/Taiwan region with Peat Burning or Agricultural Waste Burning
        has_china_region = any('China' in str(idx) or 'Taiwan' in str(idx) for idx in model.index if hasattr(idx, '__iter__'))
        has_awb_or_peat = any('Peat' in str(idx) or 'Agricultural Waste' in str(idx) for idx in model.index if hasattr(idx, '__iter__'))
        
        if has_china_region and has_awb_or_peat:
            print(f"\n{'='*70}")
            print("SPECIAL DIAGNOSTIC: China/Taiwan region with Peat/AWB sectors")
            print(f"{'='*70}")
            
            # Find the region name(s) that contain China or Taiwan
            china_regions = [idx for idx in model.index if 'China' in str(idx) or 'Taiwan' in str(idx)]
            
            for region_idx in china_regions[:3]:  # Check first 3 matching regions
                print(f"\nRegion index: {region_idx}")
                
                # Get countries in this region
                region_name = region_idx[model.index.names.index('region')] if 'region' in model.index.names else None
                if region_name:
                    try:
                        countries_in_region = context.regionmap[context.regionmap == region_name].index.get_level_values(0).unique()
                        print(f"  Countries in region: {list(countries_in_region)}")
                        
                        # Check hist_aligned for each country in this region
                        print(f"\n  Historical values for each country:")
                        for country in countries_in_region:
                            # Try to find this country in hist_aligned for this trajectory
                            country_mask = hist_aligned.index.get_level_values('country') == country
                            if 'sector' in hist_aligned.index.names:
                                sector_mask = hist_aligned.index.get_level_values('sector').str.contains('Peat|Agricultural Waste', na=False, regex=True)
                                combined_mask = country_mask & sector_mask
                                if combined_mask.any():
                                    country_data = hist_aligned[combined_mask]
                                    print(f"    {country}: {len(country_data)} rows")
                                    for idx, val in country_data.head(3).items():
                                        print(f"      {idx}: {val}")
                                else:
                                    print(f"    {country}: No Peat/AWB data found")
                            else:
                                if country_mask.any():
                                    print(f"    {country}: {hist_aligned[country_mask].iloc[0] if country_mask.sum() > 0 else 'N/A'}")
                                else:
                                    print(f"    {country}: Not in hist_aligned")
                                    
                    except Exception as e:
                        print(f"  ERROR checking countries: {e}")
            
            print(f"{'='*70}\n")

    # =========================================================================
    # DIAGNOSTIC 4: Check weights for NaN and negative values
    # =========================================================================
    if log_diagnostics:
        nan_weights = weights.isna().sum()
        negative_weights = (weights < 0).sum()
        
        if nan_weights > 0:
            print(f"\n⚠️  WARNING: {nan_weights} NaN weights detected!")
            print("    These will produce NaN in downscaled results.")
            
        if negative_weights > 0:
            print(f"\n⚠️  WARNING: {negative_weights} NEGATIVE weights detected!")
            print("    These will produce negative downscaled values from positive model values.")
            neg_w = weights[weights < 0]
            print(f"\nNegative weights:\n{neg_w.head(20)}")

        # Check if weights sum to 1 within each group
        weight_sums = weights.groupby(model.index.names, dropna=False).sum()
        bad_sums = weight_sums[(weight_sums < 0.99) | (weight_sums > 1.01)]
        if len(bad_sums) > 0:
            print(f"\n⚠️  WARNING: {len(bad_sums)} groups have weights that don't sum to ~1:")
            print(bad_sums.head(10))

    # =========================================================================
    # Apply weights to model data
    # =========================================================================
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("MULTIPLYING: model * weights")
        print(f"{'='*70}")
        
        # Check specifically for China/Taiwan with Peat/AWB before multiply
        for country in ['chn', 'hkg', 'mac', 'twn']:
            country_mask = weights.index.get_level_values('country') == country
            if country_mask.any():
                # Check if this country has Peat or AWB
                if 'sector' in weights.index.names:
                    sector_vals = weights[country_mask].index.get_level_values('sector')
                    has_peat_or_awb = any('Peat' in str(s) or 'Agricultural Waste' in str(s) for s in sector_vals)
                    if has_peat_or_awb:
                        peat_awb_mask = country_mask
                        if 'sector' in weights.index.names:
                            sector_mask = weights.index.get_level_values('sector').str.contains('Peat|Agricultural Waste', na=False, regex=True)
                            peat_awb_mask = country_mask & sector_mask
                        
                        country_weights = weights[peat_awb_mask]
                        print(f"  {country} Peat/AWB weights: {len(country_weights)} rows")
                        if len(country_weights) < 5:
                            for idx, val in country_weights.items():
                                print(f"    {idx}: weight={val:.6f}")
        print(f"{'='*70}\n")
    
    res = model.pix.multiply(weights, join="left")
    
    if log_diagnostics:
        print(f"\n{'='*70}")
        print("AFTER MULTIPLY: result = model * weights")
        print(f"{'='*70}")
        print(f"  res shape: {res.shape}")
        
        # Check specifically for China/Taiwan countries
        for country in ['chn', 'hkg', 'mac', 'twn']:
            country_mask = res.index.get_level_values('country') == country
            count = country_mask.sum()
            print(f"  {country}: {count} result rows")
            if count > 0 and count < 20:
                # Show Peat/AWB rows specifically
                if 'sector' in res.index.names:
                    sector_mask = res.index.get_level_values('sector').str.contains('Peat|Agricultural Waste', na=False, regex=True)
                    combined = country_mask & sector_mask
                    if combined.any():
                        country_res = res[combined]
                        print(f"    Peat/AWB rows: {len(country_res)}")
                        # Show first year value for each
                        first_col = country_res.columns[0]
                        for idx in country_res.index[:3]:
                            val = country_res.loc[idx, first_col]
                            print(f"      {idx}: {first_col}={val}")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # DIAGNOSTIC 5: Check result BEFORE the where() clause
    # =========================================================================
    if log_diagnostics:
        numeric_cols = res.select_dtypes(include=[np.number]).columns
        res_numeric = res[numeric_cols]
        has_negative_before_where = (res_numeric < 0).any(axis=1)
        
        if has_negative_before_where.any():
            print(f"\n{'='*70}")
            print(f"DIAGNOSTIC: {has_negative_before_where.sum()} rows have negatives BEFORE where() clause")
            print(f"{'='*70}")
            neg_before = res.loc[has_negative_before_where].head(5)
            print(f"Examples:\n{neg_before}")
    
    # Apply the where clause (original behavior)
    result = res.where(semijoin(model != 0, res.index, how="right"), 0)

    # =========================================================================
    # DIAGNOSTIC 6: Deep analysis of negative results
    # =========================================================================
    if log_diagnostics:
        # Check numeric columns only
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result_numeric = result[numeric_cols]
        
        # Find rows with any negative value
        has_negative = (result_numeric < 0).any(axis=1)
        negative_data = result.loc[has_negative]
        
        if len(negative_data) > 0:
            print(f"\n{'='*70}")
            print(f"ERROR: Downscaling produced {len(negative_data)} rows with negative values!")
            print(f"{'='*70}")
            
            # Show summary by index levels
            neg_summary = negative_data.reset_index()
            index_cols = [c for c in neg_summary.columns if c in result.index.names]
            for col in index_cols:
                unique_vals = neg_summary[col].unique()
                if len(unique_vals) <= 15:
                    print(f"  {col}: {sorted(unique_vals)}")
                else:
                    print(f"  {col}: {len(unique_vals)} unique values")
            
            # =========================================================
            # DEEP TRACE: For each negative row, trace back completely
            # =========================================================
            print(f"\n{'='*70}")
            print("DEEP TRACE: Analyzing negative results")
            print(f"{'='*70}")
            
            sample_negative = negative_data.head(5)
            for idx in sample_negative.index:
                print(f"\n--- Row: {idx} ---")
                
                # The result has 'country' in its index, model has 'region'
                # Get country from result index
                result_idx_names = list(result.index.names)
                country = idx[result_idx_names.index('country')]
                
                # Find the region for this country
                try:
                    region_info = context.regionmap.loc[country]
                    region = region_info if isinstance(region_info, str) else region_info.iloc[0]
                    print(f"  Country: {country} → Region: {region}")
                except:
                    print(f"  Country: {country} → Region: UNKNOWN")
                    continue
                
                # Get all countries in this region
                countries_in_region = context.regionmap[context.regionmap == region].index.get_level_values('country').unique()
                print(f"  Countries in region: {list(countries_in_region)}")
                
                # Get hist values for all countries in region for this trajectory
                print(f"\n  Historic values (base year {context.year}):")
                for c in countries_in_region:
                    try:
                        # Build the full index for hist lookup
                        # hist_aligned has same structure as result (with country)
                        hist_idx = tuple(
                            c if n == 'country' else idx[result_idx_names.index(n)]
                            for n in hist_aligned.index.names
                        )
                        h_val = hist_aligned.loc[hist_idx]
                        print(f"    {c}: {h_val:.6e}")
                    except Exception as e:
                        print(f"    {c}: ERROR - {e}")
                
                # Get weights for all countries in region
                print(f"\n  Calculated weights:")
                for c in countries_in_region:
                    try:
                        weight_idx = tuple(
                            c if n == 'country' else idx[result_idx_names.index(n)]
                            for n in weights.index.names
                        )
                        w_val = weights.loc[weight_idx]
                        print(f"    {c}: {w_val:.6e}")
                    except Exception as e:
                        print(f"    {c}: ERROR - {e}")
                
                # Get model values for this trajectory (regional, all years)
                print(f"\n  Model values (regional, by year):")
                try:
                    # Model has region level, we need to map from result index to model index
                    model_idx_names = list(model_full.index.names)
                    model_idx = tuple(
                        region if n == 'region' else idx[result_idx_names.index(n)] if n in result_idx_names else None
                        for n in model_idx_names
                    )
                    # Remove any None values
                    model_idx = tuple(v for v in model_idx if v is not None)
                    # Filter to just this trajectory
                    model_slice = model_full.loc[[model_idx]] if len(model_idx) == len(model_idx_names) else model_full
                    if not model_slice.empty:
                        print(f"    Years: {list(model_slice.columns[:10])}...")
                        print(f"    Values: {list(model_slice.iloc[0, :10].values)}...")
                    else:
                        print(f"    No model data found for this trajectory")
                except Exception as e:
                    print(f"    ERROR getting model values: {e}")
                
                # Show the actual negative result values
                print(f"\n  Result values (this country, by year):")
                result_row = result.loc[idx]
                neg_years = result_row[result_row < 0].head(5)
                print(f"    Negative years: {dict(neg_years)}")
                
            print(f"\n{'='*70}\n")

    # =========================================================================
    # Store debug info if requested - ALWAYS store to help debugging
    # =========================================================================
    if return_debug_info:
        print(f"[STORE] Storing debug info (call #{_CALL_COUNT})...")
        
        # Extract sectors from this call for logging
        sectors = []
        if isinstance(weights.index, pd.MultiIndex) and 'sector' in weights.index.names:
            sectors = list(weights.index.get_level_values('sector').unique())
        
        call_info = {
            'call_number': _CALL_COUNT,
            'sectors': sectors,
            'weights': weights.copy(),
            'hist_aligned': hist_aligned.copy(),
            'hist_full': hist.copy() if isinstance(hist, DataFrame) else hist,
            'model': model.copy(),
            'model_full': model_full.copy(),
            'result': result.copy(),
            'res_before_where': res.copy(),
            'context_year': context.year,
            'regionmap': context.regionmap.copy(),
        }
        _DEBUG_INFO_LIST.append(call_info)
        print(f"[OK] Debug info stored for call #{_CALL_COUNT} (sectors: {sectors})")

    return result


def patch_base_year_pattern(fill_missing_countries: bool = False, log_diagnostics: bool = True, return_debug_info: bool = False):
    """
    Monkey-patch aneris.downscaling.methods.base_year_pattern with the debug version.

    Parameters
    ----------
    fill_missing_countries : bool, default False
        If True, fill NaN values (from missing countries) with 0 before calculating weights.
    log_diagnostics : bool, default True
        If True, log detailed diagnostics about coverage and value issues.
    return_debug_info : bool, default False
        If True, store debug info (weights, hist_aligned, model) in a global dict
        that can be accessed after the run via `get_debug_info()`.

    Example
    -------
    >>> from debug_base_year_pattern import patch_base_year_pattern, get_debug_info
    >>> patch_base_year_pattern(return_debug_info=True)
    >>> # Run your workflow
    >>> debug_info = get_debug_info()
    >>> debug_info['weights']  # Access the weights
    """
    import aneris.downscaling.methods as methods
    import aneris.downscaling.core as core
    from functools import partial

    # Create a closure that captures the settings
    def patched_base_year_pattern(model, hist, context):
        return base_year_pattern_debug(
            model, hist, context,
            fill_missing_countries=fill_missing_countries,
            log_diagnostics=log_diagnostics,
            return_debug_info=return_debug_info,
        )

    # Apply the patch to the methods module
    methods.base_year_pattern = patched_base_year_pattern
    
    # ALSO patch the Downscaler._methods dictionary directly!
    # This is crucial because Downscaler uses its own cached reference
    core.Downscaler._methods = {
        "ipat_2100_gdp": partial(
            methods.intensity_convergence, convergence_year=2100, proxy_name="gdp"
        ),
        "ipat_2150_pop": partial(
            methods.intensity_convergence, convergence_year=2150, proxy_name="pop"
        ),
        "base_year_pattern": patched_base_year_pattern,  # Our debug version!
        "growth_rate": methods.growth_rate,
        "proxy_gdp": partial(methods.simple_proxy, proxy_name="gdp"),
        "proxy_pop": partial(methods.simple_proxy, proxy_name="pop"),
    }
    
    print(
        f"\n{'='*70}\n"
        f"[OK] Patched aneris.downscaling.methods.base_year_pattern\n"
        f"[OK] Patched aneris.downscaling.core.Downscaler._methods\n"
        f"  fill_missing_countries={fill_missing_countries}\n"
        f"  log_diagnostics={log_diagnostics}\n"
        f"  return_debug_info={return_debug_info}\n"
        f"{'='*70}\n"
    )


def patch_all_downscaling_methods(return_debug_info: bool = True):
    """
    Patch ALL downscaling methods to log when they are called.
    
    Use this to discover which method is actually being used.
    """
    import aneris.downscaling.methods as methods
    import aneris.downscaling.core as core
    from functools import partial, wraps
    
    global _DEBUG_INFO
    _DEBUG_INFO['method_calls'] = []
    
    def make_wrapper(original_func, method_name):
        @wraps(original_func)
        def wrapper(model, hist, context, *args, **kwargs):
            print(f"\n[DOWNSCALING] METHOD CALLED: {method_name}")
            print(f"   model shape: {model.shape}")
            print(f"   model index levels: {model.index.names}")
            print(f"   Sample model index (first 3):")
            for idx in model.index[:3]:
                print(f"      {idx}")
            
            _DEBUG_INFO['method_calls'].append({
                'method': method_name,
                'model_shape': model.shape,
                'model_index_names': list(model.index.names),
            })
            
            result = original_func(model, hist, context, *args, **kwargs)
            
            # Check for negatives in result
            if (result < 0).any().any():
                neg_count = (result < 0).sum().sum()
                print(f"   [WARNING] Result has {neg_count} negative values!")
                _DEBUG_INFO['negative_from_method'] = method_name
                _DEBUG_INFO['negative_result'] = result[result < 0].stack()
                _DEBUG_INFO['negative_model'] = model
                _DEBUG_INFO['negative_hist'] = hist
                _DEBUG_INFO['negative_context'] = context
            
            return result
        return wrapper
    
    # Wrap all methods
    wrapped_intensity_convergence = make_wrapper(methods.intensity_convergence, "intensity_convergence")
    wrapped_base_year_pattern = make_wrapper(methods.base_year_pattern, "base_year_pattern")
    wrapped_growth_rate = make_wrapper(methods.growth_rate, "growth_rate")
    wrapped_simple_proxy_gdp = make_wrapper(
        partial(methods.simple_proxy, proxy_name="gdp"), 
        "simple_proxy_gdp"
    )
    wrapped_simple_proxy_pop = make_wrapper(
        partial(methods.simple_proxy, proxy_name="pop"), 
        "simple_proxy_pop"
    )
    
    # Update the Downscaler._methods dictionary
    core.Downscaler._methods = {
        "ipat_2100_gdp": partial(wrapped_intensity_convergence, convergence_year=2100, proxy_name="gdp"),
        "ipat_2150_pop": partial(wrapped_intensity_convergence, convergence_year=2150, proxy_name="pop"),
        "base_year_pattern": wrapped_base_year_pattern,
        "growth_rate": wrapped_growth_rate,
        "proxy_gdp": wrapped_simple_proxy_gdp,
        "proxy_pop": wrapped_simple_proxy_pop,
    }
    
    print(
        f"\n{'='*70}\n"
        f"[OK] Patched ALL downscaling methods with logging wrappers!\n"
        f"  Methods: ipat_2100_gdp, ipat_2150_pop, base_year_pattern, growth_rate, proxy_gdp, proxy_pop\n"
        f"  Look for '[DOWNSCALING]' messages\n"
        f"{'='*70}\n"
    )



def get_debug_info(call_number=None, sector=None):
    """
    Get the stored debug information from base_year_pattern calls.
    
    Parameters
    ----------
    call_number : int, optional
        Specific call number to retrieve (1-indexed). If None, returns data from all calls.
    sector : str, optional
        Filter for calls containing this sector. If None, no filtering.
    
    Returns
    -------
    dict or list
        If call_number specified: dict with debug info from that call
        If call_number is None: list of dicts with debug info from all calls
        Returns empty dict/list if no debug info has been captured.
        
    Examples
    --------
    >>> get_debug_info()  # Get all calls
    >>> get_debug_info(call_number=1)  # Get first call
    >>> get_debug_info(sector='Agricultural Waste Burning')  # Get calls with AWB
    """
    if not _DEBUG_INFO_LIST:
        return {} if call_number is not None else []
    
    # Filter by sector if requested
    filtered_calls = _DEBUG_INFO_LIST
    if sector is not None:
        filtered_calls = [c for c in _DEBUG_INFO_LIST if sector in c.get('sectors', [])]
    
    # Return specific call or all calls
    if call_number is not None:
        # Find the call with this number
        for call in filtered_calls:
            if call['call_number'] == call_number:
                return call.copy()
        return {}  # Not found
    
    # Return all filtered calls
    return [c.copy() for c in filtered_calls]

def clear_debug_info():
    """Clear the stored debug info."""
    global _DEBUG_INFO_LIST
    _DEBUG_INFO_LIST = []


# Alternative: A completely rewritten function with multiple strategies
def base_year_pattern_safe(
    model: DataFrame,
    hist: Union[Series, DataFrame],
    context,
    strategy: str = "fill_missing_with_zero",
) -> DataFrame:
    """
    Safe version of base_year_pattern with multiple strategies for handling issues.

    Parameters
    ----------
    strategy : str
        How to handle missing countries / NaN values:
        - "fill_missing_with_zero": Fill NaN (missing countries) with 0, then normalize (default)
        - "drop_missing": Only use countries present in both hist and regionmap
        - "uniform_for_missing": Give missing countries equal share of what's left
        - "original": Original behavior (may produce NaN/negatives)
    """
    from pandas_indexing import semijoin

    model = model.loc[:, context.year:]
    if isinstance(hist, DataFrame):
        hist = hist.loc[:, context.year]

    hist_aligned = semijoin(hist, context.regionmap, how="right")

    if strategy == "fill_missing_with_zero":
        # Fill NaN (missing countries) with 0, then normalize
        hist_safe = hist_aligned.fillna(0)
        weights = hist_safe.groupby(model.index.names, dropna=False).transform(normalize)

    elif strategy == "drop_missing":
        # Only use countries present in both - drop NaN rows
        hist_clean = hist_aligned.dropna()
        weights = hist_clean.groupby(model.index.names, dropna=False).transform(normalize)
        # Re-align to full index, filling missing with 0
        weights = weights.reindex(hist_aligned.index, fill_value=0)

    elif strategy == "uniform_for_missing":
        # For missing countries, distribute remaining weight uniformly
        hist_filled = hist_aligned.fillna(0)
        weights = hist_filled.groupby(model.index.names, dropna=False).transform(normalize)
        
        # Where original was NaN and weight is 0, redistribute
        # This is complex - for now just use fill_missing_with_zero
        pass

    elif strategy == "original":
        # Original behavior
        weights = hist_aligned.groupby(model.index.names, dropna=False).transform(normalize)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    res = model.pix.multiply(weights, join="left")
    return res.where(semijoin(model != 0, res.index, how="right"), 0)


if __name__ == "__main__":
    # Quick test
    print("Debug module loaded. Use patch_base_year_pattern() to apply the patch.")
