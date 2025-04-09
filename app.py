
import os
from flask import Flask, render_template, jsonify, request
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import pandas as pd 

from src.prediction_logic import get_monthly_predictions_tf, get_monthly_predictions_lr, PREDICTION_YEAR

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main page."""
    months = ["all"] + [f"{PREDICTION_YEAR}-{m:02d}" for m in range(1, 13)]
    return render_template('index.html', year=PREDICTION_YEAR, months=months)

@app.route('/get_predictions', methods=['POST'])
def get_predictions_endpoint():
    """Endpoint called by JavaScript to fetch prediction data based on user selections."""
    predictions_df = None
    error = None
    results = [] 

    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Invalid request: Missing JSON data.'}), 400

        model_type = request_data.get('model_type', 'mlp') 
        selected_month = request_data.get('selected_month', 'all') 

        print(f"Received prediction request: model='{model_type}', month='{selected_month}'")

        if model_type == 'comparison':
            print("Comparison mode requested.")
            df_tf, err_tf = get_monthly_predictions_tf()
            df_lr, err_lr = get_monthly_predictions_lr()

            if err_tf and err_lr:
                error = f"Failed to get predictions for both models. MLP Error: {err_tf}. LR Error: {err_lr}"
                return jsonify({'error': error}), 500
            if err_tf:
                error = f"Failed to get MLP predictions for comparison: {err_tf}"
                return jsonify({'error': error}), 500
            if err_lr:
                error = f"Failed to get Linear Regression predictions for comparison: {err_lr}"
                return jsonify({'error': error}), 500

            if df_tf is None or df_lr is None:
                 error = "One or both prediction dataframes are None even without explicit errors."
                 return jsonify({'error': error}), 500

            print("Merging TF and LR predictions...")
            df_tf = df_tf.rename(columns={'Predicted_Total_Receipts': 'Predicted_Total_Receipts_MLP'})
            df_lr = df_lr.rename(columns={'Predicted_Total_Receipts': 'Predicted_Total_Receipts_LR'})

            predictions_df = pd.merge(df_tf, df_lr, on='Month', how='outer')
            predictions_df.fillna({'Predicted_Total_Receipts_MLP': 0, 'Predicted_Total_Receipts_LR': 0}, inplace=True)
            print("Merge complete.")

        else:
            if model_type == 'mlp':
                predictions_df, error = get_monthly_predictions_tf()
            elif model_type == 'lr':
                predictions_df, error = get_monthly_predictions_lr()
            else:
                error = f"Invalid model type specified: {model_type}"

            if error:
                print(f"Prediction error: {error}")
                if isinstance(error, FileNotFoundError) or "not found" in str(error).lower():
                     return jsonify({'error': f"Model files not found for '{model_type}'. Please ensure the model has been trained."}), 500
                return jsonify({'error': str(error)}), 500
            if predictions_df is None:
                 return jsonify({'error': f"Prediction failed for model '{model_type}' with no specific error."}), 500

        if selected_month != 'all':
            print(f"Filtering predictions for month: {selected_month}")
            predictions_df = predictions_df[predictions_df['Month'] == selected_month].copy()
            if predictions_df.empty:
                 print(f"Warning: No data found after filtering for month {selected_month}")
                 results = [] 


        if not predictions_df.empty:
            results = predictions_df.to_dict(orient='records')
            
        print(f"Returning {len(results)} prediction record(s).")
        return jsonify({'predictions': results})

    except FileNotFoundError as fnf_error:
        print(f"File Not Found Error during request processing: {fnf_error}")
        return jsonify({'error': f"Required model or data file not found: {fnf_error}"}), 500
    except Exception as e:
        print(f"Unexpected server error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    #app.run(host='0.0.0.0', port=port, debug=True) 
    app.run(host='0.0.0.0', port=port)