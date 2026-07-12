import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# -------------------------------
# Load processed feature dataset
# -------------------------------

df = pd.read_parquet("data/processed/features.parquet")

# -------------------------------
# Features and Target
# -------------------------------

feature_columns = [
    "Temperature",
    "Humidity",
    "CO2",
    "Light",
    "Temp_Humidity"
]

target_column = "Yield"

X = df[feature_columns]
y = df[target_column]

# -------------------------------
# Time-based Train/Test Split
# -------------------------------

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

# -------------------------------
# Train Linear Regression
# -------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully.")

# -------------------------------
# Predictions
# -------------------------------

pred_train = model.predict(X_train)

pred_test = model.predict(X_test)

# -------------------------------
# Evaluation Metrics
# -------------------------------

mae = mean_absolute_error(y_test, pred_test)

rmse = np.sqrt(mean_squared_error(y_test, pred_test))

r2 = r2_score(y_test, pred_test)

print("\nEvaluation Metrics")

print("--------------------------")

print(f"MAE  : {mae:.3f}")

print(f"RMSE : {rmse:.3f}")

print(f"R²   : {r2:.3f}")

# -------------------------------
# Save metrics
# -------------------------------

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

os.makedirs("reports", exist_ok=True)

with open("reports/linear_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved.")

# -------------------------------
# Save model
# -------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/linear_regression.joblib")

print("Model saved.")

# -------------------------------
# Coefficient Table
# -------------------------------

coef_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients")

print(coef_df)

# -------------------------------
# Residuals
# -------------------------------

residuals = y_test - pred_test

os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(6,5))

plt.scatter(pred_test, residuals)

plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted Yield")

plt.ylabel("Residual")

plt.title("Residuals vs Predicted")

plt.savefig(
    "reports/figures/residuals_vs_predicted.png",
    dpi=150
)

plt.close()

plt.figure(figsize=(6,5))

plt.scatter(X_test["Humidity"], residuals)

plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Humidity")

plt.ylabel("Residual")

plt.title("Residuals vs Humidity")

plt.savefig(
    "reports/figures/residuals_vs_humidity.png",
    dpi=150
)

plt.close()

print("Residual plots saved.")

# -------------------------------
# Diagnostics Markdown
# -------------------------------

diagnostics = f"""
# Linear Regression Diagnostics

## Test Metrics

- MAE : {mae:.3f}
- RMSE : {rmse:.3f}
- R² : {r2:.3f}

## Observations

- Residual plots were generated.
- Check if residuals are randomly scattered.
- Any visible curve indicates non-linearity.
- Funnel shape indicates heteroscedasticity.
- Large isolated points indicate possible outliers.

Recommendation:
If residuals are not random, try Random Forest Regression.
"""

with open(
    "reports/linear_diagnostics.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(diagnostics)

print("Diagnostics file saved.")