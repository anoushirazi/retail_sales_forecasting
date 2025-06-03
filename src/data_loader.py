"""
data_loader.py

Handles loading and initial inspection of the retail sales dataset.
"""

import pandas as pd

def load_sales_data(csv_path):
    """
    Load retail sales data from a CSV file and perform basic preprocessing.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with parsed dates and sorted by time.
    """
    df = pd.read_csv(csv_path)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df.sort_values('Date', inplace=True)

    return df


def inspect_data(df):
    """
    Perform basic inspection of the DataFrame.

    Parameters:
        df (pd.DataFrame): The loaded DataFrame.

    Returns:
        dict: Summary information including shape, nulls, and stats.
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.to_dict(),
        "describe": df.describe().to_dict()
    }
    return summary
