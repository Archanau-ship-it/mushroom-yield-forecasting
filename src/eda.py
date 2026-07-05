import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create reports folder if it doesn't exist
os.makedirs("reports/figures", exist_ok=True)

# Load cleaned dataset
df = pd.read_parquet("data/processed/02_cleaned.parquet")

print("=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)

# -------------------------
# Date Range
# -------------------------
print("\nDate Range")
print(df["Timestamp"].min(), "to", df["Timestamp"].max())

# -------------------------
# Summary Statistics
# -------------------------
print("\nSummary Statistics")
print(df.describe())

# -------------------------
# Rule Violations
# -------------------------
print("\nRule Violations")

print("Temperature < 0 :", (df["Temperature"] < 0).sum())
print("Humidity > 100 :", (df["Humidity"] > 100).sum())
print("CO2 < 0 :", (df["CO2"] < 0).sum())
print("Yield < 0 :", (df["Yield"] < 0).sum())

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(8,6))

sns.heatmap(
    df[["Temperature","Humidity","CO2","Light","Yield"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("reports/figures/correlation_heatmap.png")

plt.close()

# -------------------------
# Humidity vs Yield
# -------------------------
plt.figure(figsize=(8,6))

plt.scatter(df["Humidity"], df["Yield"])

plt.xlabel("Humidity (%)")
plt.ylabel("Yield (kg)")
plt.title("Humidity vs Yield")

plt.grid(True)

plt.savefig("reports/figures/humidity_vs_yield.png")

plt.close()

# -------------------------
# CO2 vs Yield
# -------------------------
plt.figure(figsize=(8,6))

plt.scatter(df["CO2"], df["Yield"])

plt.xlabel("CO2 (ppm)")
plt.ylabel("Yield (kg)")
plt.title("CO2 vs Yield")

plt.grid(True)

plt.savefig("reports/figures/co2_vs_yield.png")

plt.close()

print("\nFigures saved successfully!")

print("\nEDA Task Completed Successfully!")