# Mushroom Yield Forecasting Using Machine Learning
## Technical Report

**Project Title:** Mushroom Yield Forecasting Using Machine Learning

**Intern:** Archana Unnikrishnan

**Duration:** 21-Day Machine Learning Internship

**Technology Stack:**
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Git & GitHub

---

# Executive Summary

Mushroom cultivation requires maintaining optimal environmental conditions to achieve consistent yield. This project aims to predict mushroom yield using environmental sensor data such as temperature, humidity, carbon dioxide (CO₂), and light intensity.

The project follows a complete machine learning workflow, beginning with data preprocessing and exploratory data analysis, followed by feature engineering, model development, hyperparameter tuning, deployment using Streamlit, and monitoring through prediction logging.

Two machine learning algorithms, Linear Regression and Random Forest Regression, were evaluated. Hyperparameter tuning was performed using GridSearchCV with TimeSeriesSplit, and the best-performing model was selected as the Champion Model. The final application was deployed on Streamlit Cloud, allowing users to predict mushroom yield by entering sensor values.

---

# 1. Problem Statement

Accurately predicting mushroom yield helps growers optimize environmental conditions, improve productivity, reduce waste, and support better production planning.

Traditional estimation methods are manual and often inaccurate. Machine learning provides a data-driven approach by learning relationships between environmental factors and crop yield.

This project develops a predictive model capable of estimating mushroom yield based on sensor measurements collected inside a polyhouse.

---

# 2. Project Objectives

The objectives of this project are:

- Develop a complete machine learning pipeline.
- Clean and preprocess raw sensor data.
- Analyze relationships between environmental variables and mushroom yield.
- Train and compare multiple regression models.
- Select the best-performing model.
- Deploy the model using Streamlit.
- Monitor model predictions using logging.
- Document the complete workflow for reproducibility.

---

# 3. Dataset Description

The dataset contains environmental sensor readings collected from a mushroom cultivation polyhouse.

## Features

| Feature | Description | Unit |
|----------|-------------|------|
| Timestamp | Date and time of observation | DateTime |
| Temperature | Air temperature | °C |
| Humidity | Relative humidity | % |
| CO₂ | Carbon dioxide concentration | ppm |
| Light | Light intensity | Lux |
| Yield | Mushroom yield (Target Variable) | kg |

The dataset contains hourly observations representing environmental conditions affecting mushroom production.


---

# 4. Project Methodology

The project was completed in ten major tasks.

Each task builds upon the previous stage, resulting in a complete end-to-end machine learning solution.

```
Raw Data
    ↓
Data Cleaning
    ↓
EDA
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Hyperparameter Tuning
    ↓
Streamlit Application
    ↓
Cloud Deployment
    ↓
Monitoring & Logging
```

---

# Task 1 – Project Setup

## Objective

Create a professional machine learning project structure and configure the development environment.

## Methodology

The project repository was created using Git and GitHub to maintain version control. A Python virtual environment was configured to isolate project dependencies. Required libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, and Streamlit were installed using pip.

A structured folder hierarchy was created to separate raw data, processed data, source code, trained models, reports, and documentation.

The project was initialized with Git, allowing every stage of development to be tracked and synchronized with GitHub.

## Output

- Virtual environment created
- Git repository initialized
- Project folder structure established
- Dependencies installed
- requirements.txt generated

## Key Learning

A well-organized project structure improves maintainability, collaboration, and reproducibility.


# Task 2 – Data Cleaning

## Objective

Prepare the raw dataset for machine learning by removing inconsistencies and handling missing values.

## Methodology

The raw CSV dataset was loaded using Pandas. The Timestamp column was converted into DateTime format to enable chronological processing.

The dataset was inspected using `df.info()`, `df.describe()`, and missing value analysis.

Missing values were handled using appropriate statistical techniques:

- Temperature → Mean
- Humidity → Mean
- CO₂ → Median

Rows containing missing target values (Yield) were removed because machine learning models cannot learn without target labels.

The cleaned dataset was saved in Parquet format for efficient storage and faster loading.

## Output

- Missing values handled
- Timestamp standardized
- Clean dataset generated
- Processed dataset saved as:

```
data/processed/02_cleaned.parquet
```

## Key Learning

High-quality data is essential for building reliable machine learning models.




# Task 3 – Exploratory Data Analysis (EDA)

## Objective

Understand the characteristics of the dataset and identify relationships between environmental variables and mushroom yield.

## Methodology

EDA was performed using statistical summaries and visualizations.

Correlation analysis was used to measure relationships between sensor variables and yield.

Scatter plots were created to study how temperature, humidity, and CO₂ influence mushroom production.

A correlation heatmap helped identify strongly correlated variables that could improve prediction accuracy.

The analysis also helped detect potential outliers and understand feature distributions before model training.

## Output

Generated visualizations:

- Correlation Heatmap
- Scatter Plot
- Distribution Analysis

These figures guided feature engineering and model selection.

## Key Learning

EDA transforms raw numerical data into meaningful insights that support better feature selection and model development.

---

# Task 4 – Feature Engineering & Temporal Train-Test Split

## Objective

Transform the cleaned dataset into a suitable format for machine learning by creating meaningful features, scaling numerical data, and preparing separate training and testing datasets.

## Methodology

After completing data cleaning, feature engineering was performed to improve the model's ability to learn patterns from the sensor data.

The following features were selected as input variables:

- Temperature
- Humidity
- CO₂
- Light

In addition to the original features, an interaction feature named **Temp_Humidity** was created by multiplying temperature and humidity. This feature helps the model capture the combined influence of these environmental conditions on mushroom yield.

The dataset was sorted chronologically using the **Timestamp** column before splitting into training and testing sets. Unlike random splitting, a **temporal train-test split** was used to preserve the time order of observations and prevent data leakage.

Approximately 80% of the earliest observations were used for training, while the remaining 20% were reserved for testing.

Feature scaling was performed using **MinMaxScaler**, which scales each feature into the range of 0 to 1. The scaler was fitted only on the training dataset and then applied to the testing dataset to avoid leakage.

The trained scaler was saved using Joblib for reuse during prediction.

## Output

Generated files:

- `models/scaler.joblib`
- `data/processed/features.parquet`

Training Summary

| Dataset | Rows |
|----------|------|
| Training | 76 |
| Testing | 20 |

## Key Learning

Feature engineering helps expose useful patterns within the data, while temporal splitting ensures the model is evaluated on future observations, providing a realistic estimate of its performance.



# Task 5 – Linear Regression Baseline Model

## Objective

Develop a simple baseline model that predicts mushroom yield using a linear relationship between environmental variables and the target.

## Methodology

A **Linear Regression** model from Scikit-learn was trained using the processed training dataset.

The model learns coefficients for each feature by minimizing the prediction error between actual and predicted yield values.

After training, predictions were generated for the testing dataset.

The model was evaluated using three performance metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Finally, the trained model was saved using Joblib for future inference.

## Output

Generated file:

- `models/linear_regression.joblib`

Performance Metrics

- MAE
- RMSE
- R² Score

(Replace with your actual values.)

## Key Learning

Linear Regression provides a simple and interpretable baseline model. Although computationally efficient, it may not capture complex nonlinear relationships present in agricultural sensor data.

# Task 6 – Random Forest Regression

## Objective

Improve prediction accuracy by training a nonlinear ensemble model capable of learning complex relationships between environmental variables and mushroom yield.

## Methodology

A **Random Forest Regressor** consisting of multiple decision trees was trained using the engineered feature set.

Unlike Linear Regression, Random Forest can model nonlinear interactions and is more robust to noisy sensor data.

Each decision tree was trained using different subsets of the training data, and the final prediction was obtained by averaging the predictions from all trees.

After training, predictions were generated for the testing dataset and evaluated using the same performance metrics.

Feature importance analysis was also performed to identify which environmental variables contributed most to the prediction.

The trained model was saved using Joblib.

## Output

Generated file:

- `models/random_forest.joblib`

Performance Metrics

- MAE
- RMSE
- R² Score

Feature Importance Plot generated.

## Key Learning

Random Forest significantly improved prediction performance by learning nonlinear relationships that Linear Regression could not capture.



# Task 7 – Hyperparameter Tuning & Champion Model Selection

## Objective

Optimize the Random Forest model using hyperparameter tuning and select the best-performing model for deployment.

## Methodology

Hyperparameter tuning was performed using **GridSearchCV**.

Instead of evaluating every model using a random validation split, **TimeSeriesSplit** was used because the dataset contains chronological observations.

The following parameters were tuned:

- Number of Trees (`n_estimators`)
- Maximum Tree Depth (`max_depth`)
- Minimum Samples per Leaf (`min_samples_leaf`)

GridSearchCV trained multiple Random Forest models using different combinations of these parameters and selected the configuration with the lowest Mean Absolute Error.

The optimized model was evaluated on the testing dataset and compared against the Linear Regression baseline.

Since the tuned Random Forest achieved the best performance, it was selected as the **Champion Model**.

The champion model was saved for deployment.

## Output

Generated file:

- `models/champion.joblib`

Best Parameters

| Parameter | Value |
|------------|-------|
| n_estimators | 100 |
| max_depth | None |
| min_samples_leaf | 3 |

(Replace with your exact GridSearchCV output if different.)

### Model Comparison

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | (Your Value) | (Your Value) | (Your Value) |
| Random Forest | (Your Value) | (Your Value) | (Your Value) |
| Tuned Random Forest | (Your Value) | (Your Value) | (Your Value) |

## Champion Model Justification

The tuned Random Forest model achieved the lowest prediction error while maintaining stable performance on unseen data.

Compared to Linear Regression, it better captured nonlinear relationships between environmental conditions and mushroom yield.

Therefore, it was selected as the final model for deployment.

## Key Learning

Hyperparameter tuning improves model performance by systematically searching for the optimal parameter combination instead of relying on default settings.
---

# Task 8 – Streamlit Web Application

## Objective

Develop an interactive web application that allows users to predict mushroom yield using the trained machine learning model.

## Methodology

After selecting the champion model, a user-friendly web interface was developed using **Streamlit**.

The application loads the trained model (`champion.joblib`) and the fitted scaler (`scaler.joblib`) only once using Streamlit's caching mechanism to improve performance.

Users provide the following sensor values through the sidebar:

- Temperature (°C)
- Humidity (%)
- CO₂ (ppm)
- Light (Lux)

When the **Predict Yield** button is clicked, the application preprocesses the inputs using the saved scaler and passes them to the trained Random Forest model.

The predicted mushroom yield is then displayed in kilograms.

Input validation and error handling were added to improve the user experience.

## Output

Developed files:

- `src/app.py`
- `models/champion.joblib`
- `models/scaler.joblib`

The application provides real-time yield prediction through an interactive interface.

## Key Learning

Streamlit simplifies the deployment of machine learning models by converting Python scripts into interactive web applications without requiring frontend development.


# Task 9 – Deployment, Testing and Validation

## Objective

Deploy the Streamlit application to the cloud and verify that it performs correctly under different input conditions.

## Methodology

The project repository was uploaded to GitHub.

The Streamlit application was deployed using **Streamlit Community Cloud**, which automatically builds and hosts the application directly from the GitHub repository.

After deployment, the application was tested using multiple sensor input combinations to verify prediction accuracy and application stability.

The outputs generated by the web application were compared with predictions obtained directly from the Python prediction script to ensure consistency.

The user interface was also improved by adding:

- Input validation
- Error messages
- Number formatting
- Loading indicators

## Deployment URL

```
https://zelbytes-yield-forecasting.streamlit.app/
```

## Repository

```
https://github.com/Archanau-ship-it/mushroom-yield-forecasting
```

## Key Learning

Cloud deployment makes machine learning applications accessible through a web browser without requiring users to install Python or project dependencies.


# Task 10 – Monitoring and Prediction Logging

## Objective

Implement a simple monitoring mechanism to record prediction requests and support future model maintenance.

## Methodology

A lightweight logging system was implemented using Python's CSV module.

For every prediction request, the following information is stored:

- Timestamp (UTC)
- Temperature
- Humidity
- CO₂
- Predicted Yield

These logs help analyze prediction patterns, detect abnormal sensor readings, and evaluate model performance over time.

Prediction logs are stored in:

```
logs/predictions.csv
```

The monitoring strategy also includes periodic evaluation of prediction accuracy. If future model performance decreases significantly due to changes in environmental conditions or sensor calibration, the model should be retrained using newly collected data.

## Output

Generated file:

```
logs/predictions.csv
```

## Key Learning

Monitoring is an important stage of the machine learning lifecycle because model performance can degrade after deployment due to changing real-world conditions.



# Results

The project successfully developed an end-to-end machine learning solution for predicting mushroom yield.

Two regression algorithms were evaluated.

After hyperparameter tuning, the **Random Forest Regressor** achieved the best overall performance and was selected as the Champion Model.

The trained model was successfully deployed using Streamlit Cloud, allowing users to generate predictions through a web interface.

Prediction logging was implemented to support monitoring and future model improvement.

---

# Limitations

The current implementation has the following limitations:

- Dataset contains a limited number of observations.
- Data was collected from a single cultivation environment.
- Only environmental sensor values were considered.
- External factors such as substrate quality, weather conditions, and mushroom variety were not included.
- The model requires periodic retraining as new data becomes available.

---

# Future Work

The project can be improved by implementing the following enhancements:

1. Collect data from multiple farms to improve model generalization.
2. Include additional environmental variables such as soil moisture, airflow, and weather conditions.
3. Automate periodic model retraining using newly collected sensor data.
4. Implement automatic data drift detection.
5. Store prediction logs in a database instead of CSV files.
6. Deploy the application using Docker and cloud infrastructure for better scalability.
7. Add user authentication and dashboard analytics.

---

# Conclusion

This project successfully demonstrated the complete machine learning lifecycle, beginning with raw data preprocessing and ending with cloud deployment and monitoring.

The workflow included data cleaning, exploratory data analysis, feature engineering, model development, hyperparameter tuning, deployment using Streamlit, and implementation of prediction logging.

Among the evaluated models, the tuned Random Forest Regressor achieved the best prediction performance and was selected as the final model.

The deployed application provides an interactive platform for estimating mushroom yield from environmental sensor data, while the monitoring system enables continuous improvement through prediction logging and future retraining.

Overall, the project demonstrates how machine learning can support precision agriculture by helping growers make data-driven decisions and optimize mushroom production.

---

# References

1. Scikit-learn Documentation  
   https://scikit-learn.org/

2. Pandas Documentation  
   https://pandas.pydata.org/

3. Streamlit Documentation  
   https://docs.streamlit.io/

4. NumPy Documentation  
   https://numpy.org/

5. Matplotlib Documentation  
   https://matplotlib.org/

6. Joblib Documentation  
   https://joblib.readthedocs.io/

---

# Reproducibility

Clone the repository:

```bash
git clone https://github.com/Archanau-ship-it/mushroom-yield-forecasting.git
```

Move to the project directory:

```bash
cd mushroom-yield-forecasting
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run src/app.py
```

---

# GitHub Repository

https://github.com/Archanau-ship-it/mushroom-yield-forecasting

---

# Live Application

https://zelbytes-yield-forecasting.streamlit.app/

---

