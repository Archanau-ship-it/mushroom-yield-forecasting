# 🍄 Mushroom Yield Forecasting

## Project Overview

The aim of this project is to analyze environmental sensor data collected from a mushroom polyhouse and prepare it for machine learning. The project focuses on reading the raw data, cleaning it, analyzing the relationship between environmental conditions and mushroom yield, and creating visual reports.

The environmental factors used in this project are:

- Temperature (°C)
- Humidity (%)
- CO₂ (ppm)
- Light
- Mushroom Yield (kg)

---

# Project Folder Structure

```
mushroom-yield-forecasting/
│
├── data/
│   ├── raw/
│   │   └── sensor_data.csv
│   │
│   ├── processed/
│   │   ├── 02_cleaned.csv
│   │   └── 02_cleaned.parquet
│
├── src/
│   ├── ingestion.py
│   ├── cleaning.py
│   └── eda.py
│
├── reports/
│   ├── eda_summary.md
│   └── figures/
│       ├── correlation_heatmap.png
│       ├── humidity_vs_yield.png
│       └── co2_vs_yield.png
│
├── notebooks/
│
├── models/
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- PyArrow
- Git
- GitHub

---

# Environment Setup

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```


## Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn pyarrow
```

---

## Save Installed Packages

```bash
pip freeze > requirements.txt
```

---

# Task 1 : Data Ingestion

## Objective

Load the raw sensor dataset into Python and verify that the dataset is correct before performing any analysis.

## Methodology

- Read the CSV file using Pandas.
- Convert the CSV into a DataFrame.
- Display the first five rows.
- Check the number of rows and columns.
- Verify the column names.
- Check data types.
- Identify missing values.

## Output

The program displays:

- Dataset Shape
- Dataset Information
- Column Names
- Missing Values
- Sample Records

---

# Task 2 : Data Cleaning

## Objective

Improve the quality of the dataset by handling missing values and preparing clean data for further analysis.

## Methodology

- Load the raw dataset.
- Identify missing values.
- Replace missing Temperature values using the column mean.
- Replace missing Humidity values using the column mean.
- Replace missing CO₂ values using the column median.
- Remove rows with missing Yield values.
- Verify that no missing values remain.
- Save the cleaned dataset as CSV and Parquet files.

## Output

Generated files:

```
data/processed/02_cleaned.csv
```

```
data/processed/02_cleaned.parquet
```

---

# Task 3 : Exploratory Data Analysis (EDA)

## Objective

Understand the relationship between environmental conditions and mushroom yield using statistics and visualizations.

## Methodology

- Load the cleaned dataset.
- Check the date range.
- Generate summary statistics.
- Detect rule violations.
- Calculate the correlation between variables.
- Create a correlation heatmap.
- Create scatter plots:
  - Humidity vs Yield
  - CO₂ vs Yield
- Save all figures inside the reports folder.

## Output

Generated Figures

- Correlation Heatmap
- Humidity vs Yield Scatter Plot
- CO₂ vs Yield Scatter Plot

---

# Data Cleaning Strategy

| Column | Method Used | Reason |
|---------|-------------|--------|
| Temperature | Mean | Continuous numerical data |
| Humidity | Mean | Small number of missing values |
| CO₂ | Median | Less affected by outliers |
| Yield | Removed Missing Rows | Target variable should not contain missing values |

---

# Rule Validation

The following checks were performed on the cleaned dataset.

| Rule | Validation |
|-------|------------|
| Temperature ≥ 0°C | Passed |
| Humidity ≤ 100% | Passed |
| CO₂ ≥ 0 ppm | Passed |
| Yield ≥ 0 kg | Passed |

No rule violations were found in the cleaned dataset.

---

# Visualizations

The following visualizations were created.

### Correlation Heatmap

Shows the relationship between environmental variables and mushroom yield.

### Humidity vs Yield

Shows how humidity changes affect mushroom yield.

### CO₂ vs Yield

Shows how carbon dioxide concentration affects mushroom yield.

---
The cleaned dataset was loaded from:
data/processed/02_cleaned.parquet

The dataset contains the following columns:
Timestamp
Temperature
Humidity
CO2
Light
Yield
Feature Engineering

The following input features were selected:

Temperature
Humidity
CO2
Light

An additional engineered feature was created:

Temp_Humidity
Temp_Humidity = Temperature × Humidity

This feature represents the combined effect of temperature and humidity on mushroom growth. Since mushroom yield depends on both environmental conditions together, this interaction feature provides additional information to the machine learning model.

Feature Matrix (X)
The feature matrix X contains the following columns:
Temperature
Humidity
CO2
Light
Temp_Humidity

These are the input variables used for training the model.
The target variable is:
Yield
This is the value that the machine learning model will predict.

Chronological Train-Test Split
Since this is time-series data, a chronological split was used instead of a random split.

Training Data: First 80% of the dataset
Testing Data: Remaining 20% of the dataset

This approach ensures that past observations are used to predict future observations and prevents information from the future leaking into the training process.

Split Summary
Dataset	Rows
Training	76
Testing	20
Training Period
2026-01-01 00:00:00
to
2026-01-04 05:00:00
Testing Period
2026-01-04 06:00:00
to
2026-01-05 03:00:00
Feature Scaling

Feature scaling was performed using MinMaxScaler.

The scaler converts all numerical features into a range between 0 and 1.

This ensures that all features contribute equally during model training and prevents features with larger numerical values from dominating the learning process.

To avoid data leakage, the scaler was:
Fitted only on the training dataset
Applied to both training and testing datasets using the same scaling parameters
Why Data Leakage Was Avoided

The test dataset should represent unseen future data.
If the scaler is fitted using the entire dataset before splitting, it learns information from the testing data, resulting in data leakage.

To prevent this:

The dataset was first split into training and testing sets.
The scaler was fitted only on the training data.
The trained scaler was then used to transform both datasets.

This follows machine learning best practices.

Saving the Scaler

The trained MinMaxScaler was saved using Joblib.

Saved file:

models/scaler.joblib

Saving the scaler allows the same preprocessing steps to be applied to future data before making predictions.

Saving Processed Features

The processed and scaled dataset was saved as:

data/processed/features.parquet

Saving the processed features eliminates the need to repeat preprocessing before every model training step
