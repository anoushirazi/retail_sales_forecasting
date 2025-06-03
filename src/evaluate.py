"""
evaluate.py

Provides evaluation metrics for regression forecasting models.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(y_true, y_pred, model_name=None, verbose=True):
    """
    Calculate MAE, RMSE, and MAPE for model predictions.

    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        model_name (str): Optional name of the model for labeling.
        verbose (bool): Whether to print results.

    Returns:
        dict: Dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100

    result = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    if verbose:
        print(f"{model_name or 'Model'} Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

    return result
