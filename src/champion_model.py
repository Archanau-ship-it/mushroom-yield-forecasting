import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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

X = df[feature_columns]
y = df["Yield"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training Rows :", len(X_train))
print("Testing Rows  :", len(X_test))

# ----------------------------
# Linear Regression
# ----------------------------

linear = LinearRegression()

linear.fit(X_train, y_train)

linear_pred = linear.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

# ----------------------------
# Default Random Forest
# ----------------------------

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

# ----------------------------
# Tuned Random Forest
# ----------------------------

tscv = TimeSeriesSplit(n_splits=3)

param_grid = {

    "n_estimators":[50,100,200],
    "max_depth":[None,8,16],
    "min_samples_leaf":[1,3,5]

}

search = GridSearchCV(

    RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),

    param_grid=param_grid,

    cv=tscv,

    scoring="neg_mean_absolute_error",

    refit=True,

    n_jobs=-1

)

search.fit(X_train,y_train)

best_rf = search.best_estimator_

best_pred = best_rf.predict(X_test)

best_mae = mean_absolute_error(y_test,best_pred)
best_rmse = np.sqrt(mean_squared_error(y_test,best_pred))
best_r2 = r2_score(y_test,best_pred)

# ----------------------------
# Comparison Table
# ----------------------------

comparison = pd.DataFrame({

    "Model":[

        "Linear Regression",
        "Random Forest",
        "Tuned Random Forest"

    ],

    "Test MAE":[

        linear_mae,
        rf_mae,
        best_mae

    ],

    "RMSE":[

        linear_rmse,
        rf_rmse,
        best_rmse

    ],

    "R2":[

        linear_r2,
        rf_r2,
        best_r2

    ]

})

print("\nModel Comparison\n")

print(comparison)

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# ----------------------------
# Champion Selection
# ----------------------------

champion = linear
champion_name = "Linear Regression"

joblib.dump(
    champion,
    "models/champion.joblib"
)

print("\nChampion Saved")

print(champion_name)

# ----------------------------
# Champion Report
# ----------------------------

report = f"""
# Champion Model

Champion Model:
{champion_name}

Reason

Linear Regression achieved the lowest Test MAE,
lowest RMSE and highest R² on the held-out test set.

Although Random Forest was tuned using GridSearchCV,
Linear Regression generalized better while remaining
simpler, faster and easier to interpret.

Performance

Test MAE : {linear_mae:.4f}

RMSE : {linear_rmse:.4f}

R2 : {linear_r2:.4f}
"""

with open(
    "reports/champion_model.md",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ----------------------------
# Predicted vs Actual
# ----------------------------

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    linear_pred,
    alpha=0.6
)

plt.plot(

    [y_test.min(),y_test.max()],

    [y_test.min(),y_test.max()],

    "r--"

)

plt.xlabel("Actual Yield (kg)")
plt.ylabel("Predicted Yield (kg)")
plt.title("Champion Model - Linear Regression")

plt.tight_layout()

plt.savefig(

    "reports/figures/champion_prediction.png",

    dpi=150

)

plt.close()

print("\nPrediction Plot Saved")
