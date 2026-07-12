import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

df = pd.read_parquet("data/processed/02_cleaned.parquet")

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Interaction feature
df["Temp_Humidity"] = df["Temperature"] * df["Humidity"]

# ==========================
# Feature Matrix (X)
# ==========================
feature_columns = [
    "Temperature",
    "Humidity",
    "CO2",
    "Light",
    "Temp_Humidity"
]

X = df[feature_columns]

# Target
y = df["Yield"]

# ==========================
# Chronological Train/Test Split
# ==========================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==========================
# Scale (Fit only on training data)
# ==========================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# Save Scaler
# ==========================
joblib.dump(scaler, "models/scaler.joblib")

# ==========================
# Save Processed Features
# ==========================
train_df = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

train_df["Yield"] = y_train.reset_index(drop=True)

test_df = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)

test_df["Yield"] = y_test.reset_index(drop=True)

features = pd.concat([train_df, test_df], ignore_index=True)

features.to_parquet(
    "data/processed/features.parquet",
    index=False
)

# ==========================
# Summary
# ==========================

print("Training Rows :", len(X_train))
print("Testing Rows  :", len(X_test))

print("\nTraining Period")
print(df.iloc[0]["Timestamp"], "to", df.iloc[split_index-1]["Timestamp"])

print("\nTesting Period")
print(df.iloc[split_index]["Timestamp"], "to", df.iloc[-1]["Timestamp"])

print("\nFeatures Used:")
for feature in feature_columns:
    print("-", feature)

print("\nTarget: Yield")

print("\nScaler saved to:")
print("models/scaler.joblib")

print("\nProcessed features saved to:")
print("data/processed/features.parquet")