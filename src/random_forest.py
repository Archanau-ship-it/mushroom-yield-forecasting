import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


df = pd.read_parquet("data/processed/features.parquet")



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



split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training Rows :", len(X_train))
print("Testing Rows  :", len(X_test))


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")



train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)


train_mae = mean_absolute_error(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\nRandom Forest Evaluation")
print("---------------------------")
print(f"Train MAE : {train_mae:.3f}")
print(f"Test MAE  : {test_mae:.3f}")
print(f"Test RMSE : {test_rmse:.3f}")
print(f"Test R²   : {test_r2:.3f}")



metrics = {
    "Train MAE": float(train_mae),
    "Test MAE": float(test_mae),
    "Test RMSE": float(test_rmse),
    "Test R2": float(test_r2)
}

with open("reports/rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics Saved.")



joblib.dump(rf, "models/random_forest.joblib")

print("Model Saved.")


importance = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print("\nFeature Importances")
print(importance_df)

plt.figure(figsize=(7,5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "reports/figures/rf_importance.png",
    dpi=150
)

plt.close()

print("Feature Importance Plot Saved.")


tscv = TimeSeriesSplit(n_splits=5)

rf_cv = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

rf_cv = -rf_cv

print("\nTimeSeries Cross Validation")
print("------------------------------")

print("Fold MAE Scores")

for i, score in enumerate(rf_cv, start=1):
    print(f"Fold {i}: {score:.3f}")

print(f"\nMean CV MAE : {rf_cv.mean():.3f}")
print(f"Std CV MAE  : {rf_cv.std():.3f}")



linear = LinearRegression()

linear.fit(X_train, y_train)

linear_pred = linear.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        test_mae
    ],

    "RMSE": [
        linear_rmse,
        test_rmse
    ],

    "R2": [
        linear_r2,
        test_r2
    ]
})

comparison.to_csv(
    "reports/comparison_table.csv",
    index=False
)

print("\nModel Comparison")
print("-------------------")
print(comparison)



report = f"""
# Random Forest Cross Validation

## Method

TimeSeriesSplit with 5 folds was used.

## Cross Validation Results

Mean CV MAE : {rf_cv.mean():.3f}

Std CV MAE : {rf_cv.std():.3f}

## Overfitting Check

Training MAE : {train_mae:.3f}

Testing MAE : {test_mae:.3f}

Observation:

Training error is expected to be lower than testing error.
If the difference is small, the model generalizes well.
A very large difference indicates overfitting.

## Conclusion

Random Forest was compared against Linear Regression using identical train-test splits and evaluation metrics.
"""

with open(
    "reports/cv_results.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\nCV Report Saved.")

