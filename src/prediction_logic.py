
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import calendar
import time


from src.data_processor import create_features, apply_scaling, generate_future_dates
from src.train import SimpleMLP, HIDDEN_UNITS 


CHECKPOINT_SAVE_DIR = 'models/tf_checkpoint'
SCALER_FILENAME = 'scaler_params.npz'
PREDICTION_YEAR = 2022

_cached_model = None
_cached_scaler_params = None
_cached_feature_columns = None 

def load_tf_model_and_scaler(force_reload=False):
    global _cached_model, _cached_scaler_params, _cached_feature_columns
    if not force_reload and _cached_model and _cached_scaler_params and _cached_feature_columns:
        print("Using cached TF model and scaler.")
        return _cached_model, _cached_scaler_params, _cached_feature_columns

    print("Loading TF model, scaler, and feature columns from disk...")
    scaler_path = os.path.join(CHECKPOINT_SAVE_DIR, SCALER_FILENAME)
    model_dir = CHECKPOINT_SAVE_DIR

    if not os.path.exists(model_dir):
         error_msg = f"Model checkpoint directory '{model_dir}' not found."
         raise FileNotFoundError(error_msg)
    if not os.path.exists(scaler_path):
         error_msg = f"Scaler file '{scaler_path}' not found."
         raise FileNotFoundError(error_msg)
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if not latest_ckpt:
         error_msg = f"No checkpoint found in directory: {model_dir}"
         raise FileNotFoundError(error_msg)

    print(f"Found latest checkpoint: {latest_ckpt}")
    print(f"Found scaler file: {scaler_path}")

    try:
        scaler_data = np.load(scaler_path, allow_pickle=True)
       
        scaler_params = {'mean': scaler_data['scaler_mean'], 'std': scaler_data['scaler_std']}
        
        feature_columns = list(scaler_data['feature_columns'])
        num_features = len(feature_columns)
        print(f"Scaler parameters loaded. Expecting {num_features} features.")
        # print(f"Features loaded: {sorted(feature_columns)}") 
        model = SimpleMLP(num_features=num_features, hidden_units=HIDDEN_UNITS)

        _ = model(tf.zeros((1, num_features), dtype=tf.float32))

        checkpoint = tf.train.Checkpoint(model=model) 
        print(f"Attempting to restore model weights from: {latest_ckpt}")
        status = checkpoint.restore(latest_ckpt).expect_partial() 
        print("Model weights restored successfully.")

        _cached_model = model
        _cached_scaler_params = scaler_params
        _cached_feature_columns = feature_columns 

        return model, scaler_params, feature_columns

    except Exception as e:
        print(f"Error loading TensorFlow model, scaler, or feature columns: {e}")
        _cached_model, _cached_scaler_params, _cached_feature_columns = None, None, None
        import traceback
        traceback.print_exc()
        raise

def get_monthly_predictions_tf():
    
    print(f"\n--- Generating TF Predictions for {PREDICTION_YEAR} (using log1p target model) ---")
    start_pred_time = time.time()
    predictions_df = None
    error_message = None

    try:
        model, scaler_params, feature_columns_train = load_tf_model_and_scaler(force_reload=True)

        days_in_year = 366 if calendar.isleap(PREDICTION_YEAR) else 365
        future_df = generate_future_dates(start_date_str=f'{PREDICTION_YEAR}-01-01', periods=days_in_year)

        future_df_featured = create_features(future_df)
        print(f"Generated features for {len(future_df_featured)} future dates.")

        # Align and Prepare Features for Prediction
        X_future_aligned = pd.DataFrame(columns=feature_columns_train)

        cols_present = [col for col in feature_columns_train if col in future_df_featured.columns]
        X_future_aligned[cols_present] = future_df_featured[cols_present]

        X_future_aligned = X_future_aligned.fillna(0)
        X_future_aligned = X_future_aligned[feature_columns_train]

        X_future_np = X_future_aligned.values.astype(np.float32)

        print("Applying scaling to future features using training parameters...")
        X_future_scaled_np = apply_scaling(X_future_np, scaler_params) 
        X_future_tf = tf.constant(X_future_scaled_np)
        print(f"Prepared prediction input tensor with shape: {X_future_tf.shape}, dtype: {X_future_tf.dtype}")

        # 5.Model outputs log(1+y
        print("Making daily predictions (log-transformed scale) with TensorFlow model...")
        daily_predictions_log_tf = model(X_future_tf, training=False) 
        daily_predictions_log_np = daily_predictions_log_tf.numpy().flatten() 
        print(f"Log-transformed predictions generated. Shape: {daily_predictions_log_np.shape}")
        print(f"Sample log predictions: {daily_predictions_log_np[:5]}")

        daily_predictions_original_scale = np.expm1(daily_predictions_log_np)
        print("Predictions inverse-transformed back to original scale using np.expm1.")
        print(f"Sample original scale predictions: {daily_predictions_original_scale[:5]}")

        daily_predictions_original_scale[daily_predictions_original_scale < 0] = 0
        print("Daily predictions clipped at zero.")

        future_df['Predicted_Receipt_Count'] = daily_predictions_original_scale.round().astype(np.int64)

        print("Aggregating predictions by month...")
        future_df['Prediction_Month'] = future_df['Date'].dt.month
        monthly_predictions = future_df.groupby('Prediction_Month')['Predicted_Receipt_Count'].sum()

        month_labels = [f"{PREDICTION_YEAR}-{m:02d}" for m in monthly_predictions.index]
        monthly_predictions_df = pd.DataFrame({
            'Month': month_labels,
            'Predicted_Total_Receipts': monthly_predictions.values
        })

        total_pred_time = time.time() - start_pred_time
        print("\n--- Monthly Predictions Generated (TF) ---")
        print(monthly_predictions_df.to_string(index=False))
        print(f"--- Prediction Time: {total_pred_time:.2f} seconds ---")
        predictions_df = monthly_predictions_df

    except FileNotFoundError as e:
        error_message = str(e) 
        print(f"ERROR: {error_message}")
    except Exception as e:
        error_message = f"An unexpected error occurred during TF prediction: {type(e).__name__} - {e}"
        print(f"ERROR: {error_message}")
        import traceback
        traceback.print_exc()

    return predictions_df, error_message


#linear Regression part
from src.linear_regression import LinearRegressionScratch

LR_MODEL_DIR = 'models/lr_model'
LR_SCALER_FILENAME = 'scaler_params_lr.npz'
LR_MODEL_FILENAME = 'lr_model_params.npz'
_cached_lr_model_params = None
_cached_lr_scaler_params = None
_cached_lr_feature_columns = None

def load_lr_model_and_scaler(force_reload=False):
    global _cached_lr_model_params, _cached_lr_scaler_params, _cached_lr_feature_columns
    if not force_reload and _cached_lr_model_params and _cached_lr_scaler_params and _cached_lr_feature_columns:
        print("Using cached LR model and scaler.")
        return _cached_lr_model_params, _cached_lr_scaler_params, _cached_lr_feature_columns

    print("Loading LR model, scaler, and feature columns from disk...")
    scaler_path = os.path.join(LR_MODEL_DIR, LR_SCALER_FILENAME)
    model_path = os.path.join(LR_MODEL_DIR, LR_MODEL_FILENAME)

    if not os.path.exists(LR_MODEL_DIR) or not os.path.exists(scaler_path) or not os.path.exists(model_path):
        error_msg = f"LR Model directory '{LR_MODEL_DIR}' or scaler/model file not found. Please run LR training first (e.g., python -m src.train_lr)."
        print(f"ERROR: {error_msg}")
        _cached_lr_model_params, _cached_lr_scaler_params, _cached_lr_feature_columns = None, None, None
        raise FileNotFoundError(error_msg)

    try:
        scaler_data = np.load(scaler_path, allow_pickle=True)
        scaler_params = {'mean': scaler_data['scaler_mean'], 'std': scaler_data['scaler_std']}
        feature_columns = list(scaler_data['feature_columns'])
        print(f"LR Scaler parameters loaded. Expecting {len(feature_columns)} features.")

        model_data = np.load(model_path, allow_pickle=True)
        model_params = {'weights': model_data['weights'], 'bias': model_data['bias']}
        print("LR Model parameters (weights/bias) loaded.")

        _cached_lr_model_params = model_params
        _cached_lr_scaler_params = scaler_params
        _cached_lr_feature_columns = feature_columns

        return model_params, scaler_params, feature_columns

    except Exception as e:
        print(f"Error loading Linear Regression model or scaler parameters: {e}")
        _cached_lr_model_params, _cached_lr_scaler_params, _cached_lr_feature_columns = None, None, None
        import traceback
        traceback.print_exc()
        raise


def get_monthly_predictions_lr():
    print(f"\n--- Generating LR Predictions for {PREDICTION_YEAR} ---")
    start_pred_time = time.time()
    predictions_df = None
    error_message = None

    try:
        model_params, scaler_params, feature_columns_train_lr = load_lr_model_and_scaler(force_reload=True)

        days_in_year = 366 if calendar.isleap(PREDICTION_YEAR) else 365
        future_df = generate_future_dates(start_date_str=f'{PREDICTION_YEAR}-01-01', periods=days_in_year)
        future_df_featured = create_features(future_df)

        X_future_aligned_lr = pd.DataFrame(columns=feature_columns_train_lr)
        cols_present_lr = [col for col in feature_columns_train_lr if col in future_df_featured.columns]
        X_future_aligned_lr[cols_present_lr] = future_df_featured[cols_present_lr]
        X_future_aligned_lr = X_future_aligned_lr.fillna(0)
        X_future_aligned_lr = X_future_aligned_lr[feature_columns_train_lr]
        X_future_np_lr = X_future_aligned_lr.values.astype(np.float64) 

        print("Applying LR scaling to future features...")
        X_future_scaled_np_lr = apply_scaling(X_future_np_lr, scaler_params)

        print("Making daily predictions with Linear Regression model...")
        lr_model_instance = LinearRegressionScratch()
        lr_model_instance.set_params(model_params)
        daily_predictions_np_lr = lr_model_instance.predict(X_future_scaled_np_lr)

        daily_predictions_np_lr[daily_predictions_np_lr < 0] = 0 
        print("Daily LR predictions generated and clipped at zero.")

        future_df['Predicted_Receipt_Count_LR'] = daily_predictions_np_lr.round().astype(np.int64)

        print("Aggregating LR predictions by month...")
        future_df['Prediction_Month'] = future_df['Date'].dt.month
        monthly_predictions_lr = future_df.groupby('Prediction_Month')['Predicted_Receipt_Count_LR'].sum()

        month_labels = [f"{PREDICTION_YEAR}-{m:02d}" for m in monthly_predictions_lr.index]
        monthly_predictions_df = pd.DataFrame({
            'Month': month_labels,
            'Predicted_Total_Receipts': monthly_predictions_lr.values
        })

        total_pred_time = time.time() - start_pred_time
        print("\n--- Monthly Predictions Generated (LR) ---")
        print(monthly_predictions_df.to_string(index=False))
        print(f"--- LR Prediction Time: {total_pred_time:.2f} seconds ---")
        predictions_df = monthly_predictions_df

    except FileNotFoundError as e:
        error_message = str(e)
        print(f"ERROR: {error_message}")
    except Exception as e:
        error_message = f"An unexpected error occurred during LR prediction: {type(e).__name__} - {e}"
        print(f"ERROR: {error_message}")
        import traceback
        traceback.print_exc()

    return predictions_df, error_message

if __name__ == "__main__":
    #python -m src.train
    print("--- Running TF Prediction Example ---")
    df_preds_tf, err_tf = get_monthly_predictions_tf()
    if err_tf:
        print(f"\nTF Prediction failed with error: {err_tf}")

    #python -m src.train_lr
    print("\n--- Running LR Prediction Example ---")
    df_preds_lr, err_lr = get_monthly_predictions_lr()
    if err_lr:
        print(f"\nLR Prediction failed with error: {err_lr}")