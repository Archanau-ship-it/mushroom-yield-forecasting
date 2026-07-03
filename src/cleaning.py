import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/sensor_data.csv")

print("Missing Values Before Cleaning")
print(df.isnull().sum())

# Fill missing Temperature
df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())

# Fill missing Humidity
df["Humidity"] = df["Humidity"].fillna(df["Humidity"].mean())

# Fill missing CO2
df["CO2"] = df["CO2"].fillna(df["CO2"].median())

# Fill missing Light
df["Light"] = df["Light"].fillna(df["Light"].median())

# Remove rows where Yield is missing
df = df.dropna(subset=["Yield"])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Save cleaned CSV (optional)
df.to_csv("data/processed/02_cleaned.csv", index=False)

# Save cleaned Parquet
df.to_parquet("data/processed/02_cleaned.parquet", index=False)

print("\nCleaning Completed Successfully!")

print("\nFirst 10 Cleaned Rows")
print(df.head(10))