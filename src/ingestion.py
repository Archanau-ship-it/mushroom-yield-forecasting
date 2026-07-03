import pandas as pd

# Load CSV file
df = pd.read_csv("data/raw/sensor_data.csv")

print("===== DATA LOADED SUCCESSFULLY =====")
print()

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())