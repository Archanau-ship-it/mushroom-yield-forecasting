# Exploratory Data Analysis (EDA)

## Objective

To analyze the cleaned mushroom polyhouse dataset and identify relationships between environmental factors and mushroom yield.

---

## Data Quality Report

### Date Range

The dataset contains sensor readings from:

2026-01-01 to 2026-01-05

### Summary Statistics

The dataset was analyzed using descriptive statistics including:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Quartiles

### Rule Violations

- Temperature below 0°C : 0
- Humidity above 100% : 0
- CO₂ below 0 ppm : 0
- Yield below 0 kg : 0

The cleaned dataset contains no invalid values.

---

## Figures Generated

1. Correlation Heatmap
2. Humidity vs Yield Scatter Plot
3. CO₂ vs Yield Scatter Plot

---

## Insights

1. Humidity shows a positive relationship with mushroom yield.

2. CO₂ concentration has a moderate positive impact on mushroom yield.

3. Temperature remained within the suitable range for mushroom cultivation.

4. No abnormal environmental values were found after cleaning.

5. The cleaned dataset is suitable for machine learning model development.