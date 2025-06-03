"""
preprocess.py

Handles feature engineering steps for sales forecasting.
"""

import pandas as pd

def create_time_features(df):
    """
    Add year and month from the 'Date' column.
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

def add_lag_features(df, lags=[1, 2]):
    """
    Add lag features to the dataframe.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df['Total Amount'].shift(lag)
    return df

def add_rolling_features(df, window=3):
    """
    Add rolling mean and std features.
    """
    df[f'rolling_{window}'] = df['Total Amount'].rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['Total Amount'].rolling(window=window).std()
    return df

def add_holiday_indicator(df):
    """
    Mark December and January as holiday months.
    """
    df['is_holiday'] = df['Month'].isin([12, 1]).astype(int)
    return df

def fill_missing_features(df):
    """
    Fill NaNs in lag and rolling features with backward fill or statistical values.
    """
    df = df.copy()
    for col in df.columns:
        if col.startswith('lag_'):
            df[col] = df[col].bfill()
        elif col.startswith('rolling_'):
            if 'std' in col:
                df[col] = df[col].fillna(df['Total Amount'].std())
            else:
                df[col] = df[col].fillna(df['Total Amount'].mean())
    return df

def preprocess(df):
    """
    Full preprocessing pipeline.
    """
    df = create_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_holiday_indicator(df)
    df = fill_missing_features(df)
    return df
