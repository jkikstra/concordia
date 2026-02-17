"""
Quick diagnostic to check why only 'chn' appears in the China and Taiwan region
for Peat Burning sector, when hkg, mac, and twn should also be there.
"""

import pandas as pd
from pathlib import Path

# Update these paths to match your setup
HISTORY_FILE = "country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv"
history_path = Path(r"C:\Users\kikstra\Documents\GitHub\concordia\data")  # Adjust this path

# Read historical data
hist = (
    pd.read_csv(history_path / HISTORY_FILE)
    .drop(columns=['model', 'scenario'])
    .rename(columns={"region": "country"})
)

# Focus on the countries of interest and relevant sectors
countries_of_interest = ['chn', 'hkg', 'mac', 'twn']
sectors_of_interest = ['Peat Burning', 'Agricultural Waste Burning']

# Filter the data
hist_filtered = hist[
    (hist['country'].isin(countries_of_interest)) & 
    (hist['variable'].str.contains('Peat Burning|Agricultural Waste Burning', na=False, regex=True))
]

print("="*80)
print("Historical data for China, Hong Kong, Macau, Taiwan")
print("Sectors: Peat Burning, Agricultural Waste Burning")
print("="*80)

if hist_filtered.empty:
    print("\n⚠️ NO DATA FOUND for these countries and sectors!")
else:
    # Group by country and show what sectors they have
    for country in countries_of_interest:
        country_data = hist_filtered[hist_filtered['country'] == country]
        if not country_data.empty:
            print(f"\n{country.upper()}:")
            for _, row in country_data.iterrows():
                var = row['variable']
                # Check 2023 value
                value_2023 = row.get('2023', 'N/A')
                print(f"  {var}")
                print(f"    2023 value: {value_2023}")
        else:
            print(f"\n{country.upper()}: ⚠️ NO DATA")

print("\n" + "="*80)
print("\nKey insight:")
print("If hkg, mac, twn have NO DATA or ZERO values for Peat Burning,")
print("they won't appear in the downscaled results because:")
print("  1. They get weight = 0/0 = NaN during normalization")
print("  2. Or they're completely absent from the historical dataset")
print("="*80)
