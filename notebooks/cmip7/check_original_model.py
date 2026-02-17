"""Check original model values before harmonization"""
import pandas as pd

# Read model file
model_file = "c:/Users/kikstra/Documents/GitHub/concordia/results/vl_1-1-0-alpha/workflow_driver_data/vl_1-1-0-alpha_model.csv"
df = pd.read_csv(model_file)

print(f"Model file shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])}")

# Find MENA AWB rows
mena_awb = df[(df['region'].str.contains('Middle East', na=False)) & 
              (df['gas'] == 'CO') & 
              (df['sector'] == 'Agricultural Waste Burning')]

print(f"\nFound {len(mena_awb)} MENA AWB CO rows")

if len(mena_awb) > 0:
    row = mena_awb.iloc[0]
    
    print(f"\nOriginal model values (BEFORE harmonization):")
    print(f"Region: {row['region']}")
    print(f"Model: {row['model']}")
    print(f"Scenario: {row['scenario']}")
    print()
    
    years = ['2023', '2024', '2025', '2026', '2027', '2030', '2050', '2075', '2080', '2100']
    for year in years:
        if year in df.columns:
            val = row[year]
            print(f"  {year}: {val:.10f}")

