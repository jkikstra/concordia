"""
Investigate how exactly-zero scenario data can produce tiny negative values.

Possible causes:
1. Model data isn't exactly zero (very small positive/negative from harmonization)
2. Normalization (hist/sum(hist)) introduces floating-point errors
3. Multiplication of tiny values propagates errors
4. Interpolation between time points
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('c:/Users/kikstra/Documents/GitHub/concordia/notebooks/cmip7')

from debug_base_year_pattern import get_debug_info

def investigate_numerical_precision():
    """Check exact values and where precision errors are introduced."""
    debug_info = get_debug_info()
    
    if not debug_info:
        print("[ERROR] No debug info available")
        return
    
    # Get the data from call #1 (Agricultural Waste Burning)
    model_full = debug_info.get('model_full')
    weights = debug_info.get('weights')
    result = debug_info.get('res_before_where')
    
    if model_full is None or weights is None or result is None:
        print("[ERROR] Required data not available")
        return
    
    print("="*80)
    print("INVESTIGATING NUMERICAL PRECISION")
    print("="*80)
    
    # Focus on China and Taiwan region, Agricultural Waste Burning
    if 'sector' in model_full.index.names:
        awb_model = model_full[model_full.index.get_level_values('sector') == 'Agricultural Waste Burning']
        
        # Get the specific row
        china_idx = ('CO', 'Agricultural Waste Burning', 
                     'REMIND-MAgPIE 3.5-4.11|China and Taiwan', 
                     'Mt CO/yr', 'REMIND-MAgPIE 3.5-4.11', 
                     'SSP1 - Very Low Emissions')
        
        if china_idx in awb_model.index:
            row = awb_model.loc[china_idx]
            
            print(f"\nModel data for China and Taiwan, AWB:")
            print(f"Index: {china_idx}")
            
            # Check exact values for years around the transition
            years = [2023, 2024, 2025, 2026, 2027, 2030]
            
            print(f"\n{'Year':<8} {'Value':<20} {'Repr':<30} {'Exact Bits'}")
            print("-"*80)
            for year in years:
                if year in row.index:
                    val = row[year]
                    print(f"{year:<8} {val:<20.17e} {repr(val):<30} {val.hex() if hasattr(val, 'hex') else 'N/A'}")
            
            # Check if values are exactly zero
            print(f"\n{'Year':<8} {'== 0':<10} {'< 0':<10} {'abs(val)':<20}")
            print("-"*80)
            for year in years:
                if year in row.index:
                    val = row[year]
                    print(f"{year:<8} {val == 0:<10} {val < 0:<10} {abs(val):<20.17e}")
            
            # Now check weights for the countries
            print("\n" + "="*80)
            print("WEIGHTS FOR COUNTRIES")
            print("="*80)
            
            if 'sector' in weights.index.names:
                awb_weights = weights[weights.index.get_level_values('sector') == 'Agricultural Waste Burning']
                
                # Get weights for each country
                countries = ['chn', 'hkg', 'mac', 'twn']
                print(f"\n{'Country':<8} {'Weight':<20} {'Weight (full precision)':<30}")
                print("-"*80)
                for country in countries:
                    country_weights = awb_weights[awb_weights.index.get_level_values('country') == country]
                    if len(country_weights) > 0:
                        w = country_weights.iloc[0]
                        print(f"{country:<8} {w:<20.17e} {repr(w):<30}")
                
                # Check if weights sum to 1.0 exactly
                total_weight = awb_weights.sum()
                print(f"\nSum of weights: {total_weight:.20e}")
                print(f"Difference from 1.0: {abs(total_weight - 1.0):.20e}")
            
            # Now check the results
            print("\n" + "="*80)
            print("RESULTS (model * weights)")
            print("="*80)
            
            if 'sector' in result.index.names:
                awb_result = result[result.index.get_level_values('sector') == 'Agricultural Waste Burning']
                
                countries = ['chn', 'hkg', 'mac', 'twn']
                
                for country in countries:
                    country_result = awb_result[awb_result.index.get_level_values('country') == country]
                    if len(country_result) > 0:
                        print(f"\n{country.upper()}:")
                        result_row = country_result.iloc[0]
                        
                        print(f"{'Year':<8} {'Value':<25} {'Repr':<35}")
                        print("-"*80)
                        for year in years:
                            if year in result_row.index:
                                val = result_row[year]
                                print(f"{year:<8} {val:<25.17e} {repr(val):<35}")
                        
                        # Check what model * weight should give
                        if country in ['chn', 'hkg', 'mac', 'twn']:
                            country_idx = (country, 'REMIND-MAgPIE 3.5-4.11|China and Taiwan',
                                         'CO', 'Agricultural Waste Burning', 'Mt CO/yr',
                                         'REMIND-MAgPIE 3.5-4.11', 'SSP1 - Very Low Emissions')
                            if country_idx in awb_weights.index:
                                w = awb_weights.loc[country_idx]
                                print(f"\nManual calculation for year 2025:")
                                if 2025 in row.index:
                                    model_val = row[2025]
                                    result_val = result_row[2025] if 2025 in result_row.index else None
                                    expected = model_val * w
                                    print(f"  model value:     {model_val:.20e}")
                                    print(f"  weight:          {w:.20e}")
                                    print(f"  model * weight:  {expected:.20e}")
                                    print(f"  actual result:   {result_val:.20e}")
                                    if result_val is not None:
                                        diff = abs(result_val - expected)
                                        print(f"  difference:      {diff:.20e}")

    print("\n" + "="*80)
    print("POSSIBLE CAUSES OF TINY NEGATIVES")
    print("="*80)
    print("""
1. Model value is not exactly zero but a tiny negative number from:
   - Harmonization algorithm (ratio/offset methods may not preserve zero)
   - Interpolation between years
   - Upstream numerical operations

2. Weight normalization errors:
   - hist/sum(hist) may introduce precision errors
   - Sum of weights may not be exactly 1.0

3. Multiplication propagation:
   - Even if model is slightly negative, multiplication preserves sign
   - Floating point multiplication can change magnitude slightly

4. Accumulation of errors:
   - Multiple operations compound small errors
    """)

if __name__ == "__main__":
    investigate_numerical_precision()
