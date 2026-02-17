"""
Investigate whether model data contains tiny negatives or exact zeros.
"""

import sys
sys.path.append('c:/Users/kikstra/Documents/GitHub/concordia/notebooks/cmip7')

from debug_base_year_pattern import get_debug_info
import pandas as pd
import numpy as np

def check_model_values():
    """Check the model input data for exact zeros vs tiny negatives."""
    
    # Get AWB call
    awb_calls = get_debug_info(sector='Agricultural Waste Burning')
    
    if not awb_calls:
        print("[ERROR] No AWB data found. Run workflow first.")
        return
    
    awb_info = awb_calls[0]
    model_full = awb_info['model_full']
    
    print(f"Model data shape: {model_full.shape}")
    print(f"\n{'='*70}")
    print("CHECKING MODEL VALUES")
    print(f"{'='*70}")
    
    # Filter for AWB
    if 'sector' in model_full.index.names:
        awb_model = model_full[model_full.index.get_level_values('sector') == 'Agricultural Waste Burning']
        
        print(f"\nAWB model rows: {len(awb_model)}")
        print(f"AWB model regions: {list(awb_model.index.get_level_values('region').unique())}")
        
        # Check each region
        for idx, row in awb_model.iterrows():
            region = idx[2] if len(idx) > 2 else 'unknown'
            
            print(f"\n{'='*70}")
            print(f"Region: {region}")
            print(f"{'='*70}")
            
            # Analyze value distribution
            all_values = row.values
            positive = all_values[all_values > 0]
            zeros = all_values[all_values == 0]
            negatives = all_values[all_values < 0]
            tiny_negatives = negatives[negatives > -1e-10]
            
            print(f"Total years: {len(all_values)}")
            print(f"Positive values: {len(positive)}")
            print(f"Exact zeros: {len(zeros)}")
            print(f"Negative values: {len(negatives)}")
            if len(tiny_negatives) > 0:
                print(f"  Tiny negatives (> -1e-10): {len(tiny_negatives)}")
                print(f"  Range: [{negatives.min():.6e}, {negatives.max():.6e}]")
            
            # Show transition years
            print(f"\nValues around transition:")
            transition_years = [2023, 2024, 2025, 2026, 2027, 2030, 2050]
            for year in transition_years:
                if year in row.index:
                    val = row[year]
                    if val == 0:
                        status = "EXACT ZERO"
                    elif abs(val) < 1e-10:
                        status = f"TINY ({'neg' if val < 0 else 'pos'})"
                    else:
                        status = "normal"
                    print(f"  {year}: {val:.15e} ({status})")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("If you see 'TINY neg' values in model data, that's the source.")
    print("If you see only 'EXACT ZERO', then negatives are created elsewhere.")

if __name__ == "__main__":
    check_model_values()
