import pandas as pd
import numpy as np
import os
from typing import Tuple

def load_hotel_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def save_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train = X_train.copy()
    train['is_canceled'] = y_train
    
    val = X_val.copy()
    val['is_canceled'] = y_val
    
    test = X_test.copy()
    test['is_canceled'] = y_test
    
    train.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False, engine='fastparquet')
    val.to_parquet(os.path.join(output_dir, 'val.parquet'), index=False, engine='fastparquet')
    test.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False, engine='fastparquet')
    
    print(f"Splits saved to {output_dir}")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

def load_splits(data_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_parquet(os.path.join(data_dir, 'train.parquet'), engine='fastparquet')
    val = pd.read_parquet(os.path.join(data_dir, 'val.parquet'), engine='fastparquet')
    test = pd.read_parquet(os.path.join(data_dir, 'test.parquet'), engine='fastparquet')
    
    X_train = train.drop('is_canceled', axis=1)
    y_train = train['is_canceled']
    
    X_val = val.drop('is_canceled', axis=1)
    y_val = val['is_canceled']
    
    X_test = test.drop('is_canceled', axis=1)
    y_test = test['is_canceled']
    
    print("Splits loaded successfully.")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test