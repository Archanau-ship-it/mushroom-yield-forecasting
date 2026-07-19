# 🍄 Mushroom Yield Forecasting

> An end-to-end Machine Learning application for predicting mushroom yield using environmental sensor data, featuring data preprocessing, model training, interactive visualization, and cloud deployment with Streamlit.

---

## 📖 Overview

Mushroom cultivation requires maintaining an optimal growing environment to achieve consistent and high-quality yield. Even small fluctuations in temperature, humidity, or carbon dioxide concentration can significantly impact production.

This project develops a complete machine learning pipeline that predicts mushroom yield from environmental sensor readings. It includes data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, deployment through Streamlit, and lightweight monitoring for production readiness.

The application enables growers to estimate expected yield based on real-time environmental conditions, allowing better operational planning and environmental control.

---

## 🚀 Features

- End-to-end machine learning workflow
- Automated data preprocessing pipeline
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Linear Regression baseline model
- Random Forest regression model
- Hyperparameter tuning using GridSearchCV
- Champion model selection
- Interactive Streamlit web application
- Prediction validation and sanity checks
- Friendly error handling
- Lightweight prediction monitoring
- Cloud deployment ready

---

# Project Architecture

```
                Raw Sensor Data
                       │
                       ▼
             Data Cleaning Pipeline
                       │
                       ▼
             Feature Engineering
                       │
                       ▼
               Model Training
                       │
                       ▼
           Hyperparameter Tuning
                       │
                       ▼
            Champion Model Saved
                       │
                       ▼
              Streamlit Web App
                       │
                       ▼
             Yield Prediction
```

---

# Dataset

The dataset contains environmental sensor readings collected from a mushroom growing environment.

### Input Features

- Temperature (°C)
- Humidity (%)
- CO₂ (ppm)
- Light Intensity

### Target Variable

- Mushroom Yield (kg)

---

# Machine Learning Pipeline

The project follows a production-oriented workflow.

### 1. Data Collection

Sensor readings are loaded from CSV files.

---

### 2. Data Cleaning

The preprocessing pipeline performs:

- Missing value handling
- Datetime conversion
- Data type correction
- Duplicate removal
- Invalid record filtering

---

### 3. Exploratory Data Analysis

The project includes:

- Statistical summaries
- Correlation analysis
- Scatter plots
- Distribution plots
- Missing value analysis

---

### 4. Feature Engineering

Additional predictive features are created, including:

- Temperature × Humidity interaction
- Time-aware preprocessing
- Feature scaling using MinMaxScaler

---

### 5. Model Training

Multiple regression models were evaluated.

- Linear Regression
- Random Forest Regressor

---

### 6. Hyperparameter Optimization

Random Forest was optimized using:

- GridSearchCV
- TimeSeriesSplit Cross Validation

The best-performing model is selected as the Champion Model.

---

## 🖥️ Streamlit Application

The project includes an interactive dashboard where users can:

- Enter environmental conditions
- Predict mushroom yield
- View prediction instantly
- Receive validation messages
- Handle invalid inputs gracefully

The application includes:

- Loading spinner
- Friendly error handling
- Formatted prediction output
- Prediction sanity checks

---

## 📂 Project Structure

```
mushroom-yield-forecasting/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── champion.joblib
│   └── scaler.joblib
│
├── reports/
│   └── figures/
│
├── src/
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── champion_model.py
│   └── app.py
│
└── tests/
    └── test_predict.py
```

---

# Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

### Development Tools

- VS Code
- Git
- GitHub

---

# Installation

Clone the repository

```bash
git clone https://github.com/Archanau-ship-it/mushroom-yield-forecasting.git
```

Move into the project

```bash
cd mushroom-yield-forecasting
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Run the Streamlit app

```bash
streamlit run app.py
```

or

```bash
streamlit run src/app.py
```

(depending on your project structure)

---

# Model Performance

The project evaluates models using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The best-performing model is automatically selected and saved as the Champion Model.

---

# Testing

The project includes prediction validation tests.

Example:

```bash
pytest
```

Tests verify:

- Prediction returns float values
- Predictions remain within reasonable ranges
- Application output matches prediction pipeline
- Model artifacts load correctly

---

# Deployment

The application is designed for deployment on Streamlit Community Cloud.

Deployment includes:

- Version-pinned dependencies
- Model artifact loading
- Production-ready prediction interface
- Friendly error handling


# Monitoring

The application supports lightweight monitoring by logging:

- Timestamp
- Sensor Inputs
- Predicted Yield

Monitoring helps identify unusual prediction patterns and determine when retraining may be required.

---

# Future Improvements

- Real-time IoT sensor integration
- Weather API integration
- Deep Learning models
- Time-series forecasting
- Automated model retraining
- Docker containerization
- CI/CD pipeline
- Database-backed prediction history
- User authentication
- Mobile-friendly dashboard

---

# Author

**Archana Unnikrishnan**

Computer Science Engineering Student

GitHub: https://github.com/Archanau-ship-it

# License

This project is developed for educational and research purposes.