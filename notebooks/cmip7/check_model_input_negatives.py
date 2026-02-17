"""
Check if the model input data (regional scenarios) contains negative values
that are causing the downstream negative issues.
"""

import pandas as pd
import sys
sys.path.append('c:/Users/kikstra/Documents/GitHub/concordia/notebooks/cmip7')

# Import the debug module to get stored info
from debug_base_year_pattern import get_debug_info

def check_model_negatives():
    """Check the model input data for negative values."""
    debug_info = get_debug_info()
    
    if not debug_info or 'model_full' not in debug_info:
        print("[ERROR] No model_full data available in debug_info")
        print(f"Available keys: {list(debug_info.keys()) if debug_info else 'None'}")
        return
    
    model_df = debug_info['model_full']
    print(f"Model data shape: {model_df.shape}")
    print(f"Model index names: {model_df.index.names}")
    print(f"Model columns (years): {list(model_df.columns[:5])} ... {list(model_df.columns[-5:])}")
    
    # Check for negative values
    neg_mask = (model_df < 0).any(axis=1)
    
    if not neg_mask.any():
        print("\n[OK] No negative values found in model data")
        return
    
    neg_rows = model_df[neg_mask]
    print(f"\n[WARNING] Found {len(neg_rows)} rows with negative values in model data!")
    
    # Analyze by sector
    if 'sector' in neg_rows.index.names:
        print("\nSectors with negatives:")
        sectors = neg_rows.index.get_level_values('sector').unique()
        for sector in sectors:
            sector_neg = neg_rows[neg_rows.index.get_level_values('sector') == sector]
            print(f"  {sector}: {len(sector_neg)} rows")
    
    # Focus on Agricultural Waste Burning
    if 'sector' in neg_rows.index.names:
        awb_neg = neg_rows[neg_rows.index.get_level_values('sector') == 'Agricultural Waste Burning']
        if len(awb_neg) > 0:
            print(f"\n{'='*70}")
            print("AGRICULTURAL WASTE BURNING - MODEL INPUT NEGATIVES")
            print(f"{'='*70}")
            print(f"\nRows with negatives: {len(awb_neg)}")
            
            # Show which years have negatives
            for idx, row in awb_neg.iterrows():
                neg_years = row[row < 0].index.tolist()
                pos_years = row[row > 0].index.tolist()
                zero_years = row[row == 0].index.tolist()
                
                print(f"\nRow: {idx}")
                print(f"  Positive years: {len(pos_years)} (e.g., {pos_years[:5] if pos_years else 'none'})")
                print(f"  Zero years: {len(zero_years)} (e.g., {zero_years[:5] if zero_years else 'none'})")
                print(f"  Negative years: {len(neg_years)} (e.g., {neg_years[:5] if neg_years else 'none'})")
                
                # Show values for a few years
                sample_years = [2023, 2024, 2025, 2026, 2030, 2050, 2100]
                available_years = [y for y in sample_years if y in row.index]
                if available_years:
                    print(f"  Sample values:")
                    for year in available_years:
                        val = row[year]
                        print(f"    {year}: {val:.6e}")
    
    # Check when negatives start appearing
    print(f"\n{'='*70}")
    print("YEAR-BY-YEAR ANALYSIS")
    print(f"{'='*70}")
    for year in sorted(model_df.columns):
        year_col = model_df[year]
        n_neg = (year_col < 0).sum()
        n_zero = (year_col == 0).sum()
        n_pos = (year_col > 0).sum()
        
        if n_neg > 0 or year in [2023, 2024, 2025]:
            print(f"{year}: positive={n_pos}, zero={n_zero}, negative={n_neg}")
            if n_neg > 0:
                # Show which sectors
                neg_this_year = model_df[model_df[year] < 0]
                if 'sector' in neg_this_year.index.names:
                    sectors_neg = neg_this_year.index.get_level_values('sector').unique()
                    print(f"  Sectors: {list(sectors_neg)}")

if __name__ == "__main__":
    check_model_negatives()
