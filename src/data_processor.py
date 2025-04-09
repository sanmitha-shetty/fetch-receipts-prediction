
import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads daily receipt data from a CSV file where the header might start with '#'.
    """
    try:
    
        df = pd.read_csv(filepath, skipinitialspace=True)

        df.columns = df.columns.str.lstrip('#').str.strip()

        if 'Date' not in df.columns:
             raise ValueError("Cleaned CSV headers must contain 'Date' column.")
        if 'Receipt_Count' not in df.columns:
             raise ValueError("Cleaned CSV headers must contain 'Receipt_Count' column.")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        df['Receipt_Count'] = pd.to_numeric(df['Receipt_Count'], errors='coerce')
        df.dropna(subset=['Receipt_Count'], inplace=True) 
        df['Receipt_Count'] = df['Receipt_Count'].astype(np.float32) 

        print(f"Loaded data with {len(df)} rows. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except ValueError as ve: 
        print(f"Error processing data columns: {ve}")
        raise
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        raise


def create_features(df):
    """Creates time-based features from the Date column."""
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
   
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days

    
    df = pd.get_dummies(df, columns=['month', 'day_of_week'], drop_first=False) 
    # print(f"Created features. New shape: {df.shape}") 
    return df

def scale_features(X_train, X_test=None):
    """
    Standardizes features (mean=0, std=1) based on training data.
    Implements scaling from scratch using numpy. Ensures float32 output.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray, optional): Testing features. Defaults to None.

    Returns:
        tuple: (Scaled X_train, Scaled X_test (or None), scaler_params)
               scaler_params is a dict {'mean': mean, 'std': std}
    """
    print("Scaling features...")
    
    X_train = X_train.astype(np.float32)
    if X_test is not None:
        X_test = X_test.astype(np.float32)

    mean = np.mean(X_train, axis=0, dtype=np.float32)
    std = np.std(X_train, axis=0, dtype=np.float32)

    std_safe = np.where(std == 0, 1.0, std).astype(np.float32)

    scaler_params = {'mean': mean, 'std': std_safe} 

    # Scale training data
    X_train_scaled = (X_train - mean) / std_safe
    print("Training features scaled.")

    X_test_scaled = None
    if X_test is not None:
        # Scale test data 
        X_test_scaled = (X_test - mean) / std_safe
        print("Test features scaled using training parameters.")

    return X_train_scaled, X_test_scaled, scaler_params


def apply_scaling(X, scaler_params):
    """Applies pre-calculated scaling parameters to new data. Ensures float32."""
    if not scaler_params or 'mean' not in scaler_params or 'std' not in scaler_params:
        raise ValueError("Invalid scaler_params provided.")

    X = X.astype(np.float32)

    mean = scaler_params['mean'].astype(np.float32)
    std_safe = scaler_params['std'].astype(np.float32) 

    return (X - mean) / std_safe

def generate_future_dates(start_date_str, periods):
    """Generates future dates for prediction."""
    future_dates = pd.date_range(start=start_date_str, periods=periods, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    # print(f"Generated {len(future_df)} future dates starting from {start_date_str}.") 
    return future_df

def get_monthly_actuals(filepath, year):
    """
    Loads data and aggregates receipt counts by month for a specific year.

    Args:
        filepath (str): Path to the data CSV file.
        year (int): The year to filter data for.

    Returns:
        pandas.DataFrame: DataFrame with ['Month', 'Actual_Total_Receipts'] or None if error.
        string: Error message or None.
    """
    print(f"--- Getting Actual Monthly Data for {year} ---")
    try:
        df = load_data(filepath) 
        
        df_year = df[df['Date'].dt.year == year].copy()
        if df_year.empty:
            return None, f"No data found for the year {year} in {filepath}"
        
        df_year['Actual_Month'] = df_year['Date'].dt.month
        monthly_actuals = df_year.groupby('Actual_Month')['Receipt_Count'].sum()


        
        month_labels = [f"{year}-{m:02d}" for m in monthly_actuals.index]
        monthly_actuals_df = pd.DataFrame({
            'Month': month_labels,
            'Actual_Total_Receipts': monthly_actuals.values 
        })
        print(f"Successfully aggregated actuals for {year}.")
        return monthly_actuals_df, None

    except FileNotFoundError:
        err = f"Error: File not found at {filepath}"
        print(err)
        return None, err
    except Exception as e:
        err = f"Error processing actuals for {year}: {e}"
        print(err)
        import traceback
        traceback.print_exc()
        return None, err