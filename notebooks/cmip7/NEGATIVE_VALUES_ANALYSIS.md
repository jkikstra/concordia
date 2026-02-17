# Analysis: Negative Values in Downscaled Emissions

## Summary

The negative values in downscaled emissions are **NOT caused by the downscaling algorithm** - they originate from the **model input data** (regional scenarios) containing negative or near-zero values.

## Evidence

### 1. Downscaling Algorithm Works Correctly

From the debug output for Agricultural Waste Burning (call #1):

- **All countries survive the pipeline:**
  - chn: 2 rows (AWB + Grassland Burning)
  - hkg: 2 rows  
  - mac: 2 rows
  - twn: 2 rows

- **Weights are calculated correctly:**
  ```
  chn: 0.9938 (99.38%)
  hkg: 0.000026 (0.0026%)
  mac: 0.00000035 (0.000035%)
  twn: 0.0062 (0.62%)
  ```

- **Multiplication produces correct results:**
  - 2023: All positive values
  - 2024: All positive values

### 2. Negatives Appear from the Model Data

The diagnostic shows values over time for the China and Taiwan region:

```
Year 2023: 1.500 Mt CO/yr (positive)
Year 2024: 0.750 Mt CO/yr (positive)
Year 2025: -3.3e-14 Mt CO/yr (NEGATIVE!)
Year 2026: -3.2e-14 Mt CO/yr (negative)
...
Year 2100: -1.3e-14 Mt CO/yr (negative)
```

**Key observation:** The values from 2025 onwards are essentially zero but slightly negative (order of magnitude 10^-14 to 10^-20).

### 3. All Countries Show the Same Pattern

Because all countries use the same regional scenario value multiplied by their respective weights:

- `result[country] = model[region] × weight[country]`

When `model[region]` is negative (even tiny), ALL countries in that region get negative results:

```
chn: -3.26e-14 (99.38% of regional value)
hkg: -8.58e-19 (0.0026% of regional value)
mac: -1.13e-20 (0.000035% of regional value)
twn: -2.02e-16 (0.62% of regional value)
```

## Root Cause

The regional scenario data in the model input contains:
1. Positive values for early years (2023-2024)
2. Values that transition to essentially zero around 2025
3. Due to floating-point precision, these "zeros" become tiny negative values
4. These tiny negatives propagate through downscaling to all countries

## Why Only 'chn' Appears for Peat Burning

For Peat Burning specifically (call #3), the diagnostic shows:

```
BEFORE SEMIJOIN:
  chn: in hist=True, in regionmap=True
  hkg: in hist=False, in regionmap=False
  mac: in hist=False, in regionmap=False
  twn: in hist=False, in regionmap=False
```

**Explanation:** Hong Kong, Macau, and Taiwan do not have Peat Burning emissions in the historical inventory, so they are not in the region mapping for this sector. This is correct behavior - only China has peat burning emissions.

## Recommended Actions

### 1. Check Source Data
Verify the model input files for Agricultural Waste Burning in the "China and Taiwan" region:
- Why does the scenario go to near-zero after 2024?
- Is this intentional policy (phasing out agricultural burning)?
- Are the tiny negative values due to numerical precision issues in the scenario model?

### 2. Apply Post-Processing
If these tiny negatives are numerical artifacts, consider:
```python
# Option 1: Floor at zero
result = result.clip(lower=0)

# Option 2: Set values below threshold to zero
threshold = 1e-10
result = result.where(result.abs() > threshold, 0)
```

### 3. Investigate Other Regions
The diagnostic shows negatives also appear in:
- Middle East and North Africa
- Non-EU28 Europe

Check if these also transition to near-zero values in 2025.

## Verification Script

Run the diagnostic script to examine model input data:
```python
python check_model_input_negatives.py
```

This will show:
- Which rows in model data have negatives
- Which years contain negatives
- Sample values showing the transition from positive to negative

## Conclusion

**The downscaling algorithm is working as designed.** The negative values are an artifact of:
1. Regional scenarios that go to near-zero values
2. Floating-point precision creating tiny negative values instead of exact zeros
3. These being correctly propagated through the downscaling multiplication

The fix should be applied either:
- At the source (scenario generation)
- Or as post-processing (clip to zero)

Not in the downscaling algorithm itself, which is functioning correctly.
