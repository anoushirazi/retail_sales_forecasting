"""
utils.py

Utility functions for sales forecasting project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import joblib

def plot_forecast(dates, actual, predicted, title="Forecast vs Actual", model_name="Model"):
    """
    Plot actual vs predicted values for a forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', marker='o')
    plt.plot(dates, predicted, label=f'{model_name} Prediction', marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_model(model, path):
    """
    Save a trained model to disk using joblib.
    """
    joblib.dump(model, path)
    print(f"✅ Model saved to: {path}")

def load_model(path):
    """
    Load a model from disk using joblib.
    """
    return joblib.load(path)

def create_future_dates(start_date, periods, freq='W'):
    """
    Create a future datetime range.
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)
