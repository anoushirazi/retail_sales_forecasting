"""
train.py

Main training pipeline for retail sales forecasting.
"""

import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from data_loader import load_sales_data
from preprocess import preprocess

# --- Config ---
CSV_PATH = "data/Target_Retail_Sales_Forecasting.csv"
MODEL_OUTPUT_PATH = "models/tuned_xgb_model.pkl"
SPLIT_DATE = "2023-10-01"

# --- Load & Preprocess ---
df = load_sales_data(CSV_PATH)
df = preprocess(df)

# --- Split ---
train = df[df['Date'] < SPLIT_DATE]
test = df[df['Date'] >= SPLIT_DATE]

X_train = train[['lag_1', 'lag_2', 'rolling_3', 'rolling_std_3', 'is_holiday']]
y_train = train['Total Amount']
X_test = test[['lag_1', 'lag_2', 'rolling_3', 'rolling_std_3', 'is_holiday']]
y_test = test['Total Amount']

# --- Train Tuned XGBoost (best from GridSearchCV) ---
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    objective='reg:squarederror'
)

xgb_model.fit(X_train, y_train)

# --- Evaluate ---
preds = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
mape = (abs((y_test - preds) / y_test).mean()) * 100

print(f"Tuned XGBoost Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# --- Save ---
joblib.dump(xgb_model, MODEL_OUTPUT_PATH)
print(f"✅ Model saved to: {MODEL_OUTPUT_PATH}")
