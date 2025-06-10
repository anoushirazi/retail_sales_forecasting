# 🛍️ Retail Sales Forecasting

> A complete, deployment-ready time series forecasting solution using statistical and machine learning models, optimized for retail demand prediction.

---

## 📘 Project Overview

This project presents an end-to-end **retail sales forecasting system**, designed using real-world transactional data to help businesses with smarter **inventory planning**, **resource allocation**, and **strategic decisions**.

It covers:
- Exploratory Data Analysis (EDA)
- Feature Engineering (lags, rolling stats, holiday indicators)
- Time Series Forecasting (ARIMA, SARIMA)
- Machine Learning Models (Prophet, Random Forest, XGBoost)
- Hyperparameter Tuning
- Model Comparison & Evaluation
- Exporting model for deployment

---

## 📊 Dataset

- 📅 **Time Frame**: Daily transactions aggregated weekly
- 🧾 **Features**: Customer demographics, product categories, quantity, unit price, and total amount
- 🧼 **Preprocessing**:
  - Converted `Date` to datetime
  - Aggregated by week
  - Created lag features and rolling windows
  - Added holiday flags

---

## 🧠 Models Used

| Model           | Type            | Libraries                         |
|----------------|------------------|----------------------------------|
| ARIMA, SARIMA  | Statistical       | `statsmodels`                    |
| Prophet        | Time-series ML    | `Prophet`                        |
| Random Forest  | Ensemble ML       | `scikit-learn`                   |
| XGBoost        | Gradient Boosting | `xgboost`, `scikit-learn`        |

---

## ⚙️ Feature Engineering Highlights

- **Lag features**: 1-week, 2-week
- **Rolling stats**: 3-week rolling mean & std
- **Date-based features**: Year, month
- **Holiday indicator**: December & January

---

## 🧪 Model Evaluation

| Model          | MAE     | RMSE    | MAPE     |
|----------------|---------|---------|----------|
| ARIMA          | 2344.77 | 3488.09 | —        |
| SARIMA         | 7306.00 | 7780.83 | —        |
| Prophet        | 1610.31 | 1969.90 | 23.54%   |
| Random Forest  | 1995.69 | 2467.72 | 35.07%   |
| **Tuned XGBoost** | **1786.06** | **2264.95** | **25.39%** |

➡️ **Tuned XGBoost** gave the most balanced performance across all metrics, making it the final deployed model.

---

## 📦 Deployment Readiness

- ✅ Trained model saved using `joblib`
- ✅ Forecasts 30+ days ahead
- ✅ Modular code structure for easy integration
- ✅ Visual reports & performance plots included

---

## 📈 Visual Output Samples

- Time series plots
- Forecast vs. actual (per model)
- MAE/RMSE/MAPE comparisons
- Recursive 30-day forecast

---

## 💼 Why This Project Matters

This project showcases the full lifecycle of a real-world data science problem:
- Thoughtful data wrangling
- Creative feature engineering
- Comparative modeling
- Business-aligned evaluation
- Reproducible, interpretable code

It reflects skills that are directly aligned with roles in:
- 🏪 Retail Analytics
- 📦 Inventory Optimization
- 📈 Demand Forecasting
- 🧠 ML Engineering & Deployment

---

## 🛠️ Tools & Libraries

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Prophet, XGBoost, Scikit-Learn, Statsmodels
- Joblib for model persistence

---

## 📂 Project Structure

- `data/`  
  Contains input data files (excluded from Git for size/privacy).

- `notebooks/`  
  Jupyter notebooks used for exploration, modeling, and visualization.

- `src/`  
  Source code for data preprocessing, feature engineering, and model training.

- `models/`  
  Trained model files (e.g., `joblib` `.pkl`) saved for deployment.

- `visuals/`  
  Charts, plots, and visual assets generated during analysis.

- `requirements.txt`  
  List of required Python libraries for environment setup.

- `README.md`  
  Project documentation (you're reading it)..


