# src/train_lr.py
import pandas as pd
import numpy as np
import os
import time

from src.data_processor import load_data, create_features, scale_features
from src.linear_regression import LinearRegressionScratch

DATA_FILEPATH = 'data/data_daily.csv'
# seperate dir for lr
MODEL_SAVE_DIR = 'models/lr_model'
SCALER_FILENAME = 'scaler_params_lr.npz'
MODEL_FILENAME = 'lr_model_params.npz'

FEATURE_COLS_TO_DROP = ['Date', 'Receipt_Count', 'year'] 
TARGET_COL = 'Receipt_Count'

LEARNING_RATE = 0.01
ITERATIONS = 2000 

def train_lr_model():
    """Loads data, preprocesses, trains the Linear Regression model, and saves it."""
    print("--- Starting Linear Regression Model Training ---")
    start_time = time.time()

    try:
        df = load_data(DATA_FILEPATH)
    except Exception as e:
        print(f"Failed to load data. Exiting. Error: {e}")
        return

    df_featured = create_features(df)
    feature_columns = [col for col in df_featured.columns if col not in FEATURE_COLS_TO_DROP]
    print(f"Using {len(feature_columns)} features: {sorted(feature_columns)}")

    X = df_featured[feature_columns].values.astype(np.float64) 
    y = df_featured[TARGET_COL].values.astype(np.float64)

    
    X_scaled, _, scaler_params = scale_features(X) 
    num_features = X_scaled.shape[1]

   
    print(f"Training LinearRegressionScratch (LR={LEARNING_RATE}, Iter={ITERATIONS})")
    lr_model = LinearRegressionScratch(learning_rate=LEARNING_RATE, iterations=ITERATIONS, verbose=True)
    lr_model.fit(X_scaled, y) 

    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    scaler_path = os.path.join(MODEL_SAVE_DIR, SCALER_FILENAME)
    np.savez(scaler_path,
             scaler_mean=scaler_params['mean'],
             scaler_std=scaler_params['std'],
             feature_columns=np.array(feature_columns)) 
    print(f"LR Scaler parameters saved to: {scaler_path}")

    lr_params = lr_model.get_params()
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    np.savez(model_path,
             weights=lr_params['weights'],
             bias=lr_params['bias'])
    print(f"LR Model parameters saved to: {model_path}")

    total_training_time = time.time() - start_time
    print(f"--- Total LR Training Time: {total_training_time:.2f} seconds ---")

if __name__ == "__main__":
    train_lr_model()