import joblib
import pandas as pd
import os
import numpy as np
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# --- SETUP ---
model = None
model_columns = []
csv_path = os.path.join(settings.BASE_DIR, 'creditcard.csv') # Path to real dataset

try:
    base_path = settings.BASE_DIR
    model_path = os.path.join(base_path, 'fraud_model_pipeline.pkl')
    columns_path = os.path.join(base_path, 'model_columns.pkl')

    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
    else:
        print("⚠️ Model files missing.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- ENDPOINTS ---

@api_view(['POST'])
def predict(request):
    if model is None:
        return Response({'error': 'Model not active'}, status=503)

    try:
        input_data = request.data
        df = pd.DataFrame([input_data])
        df_clean = pd.DataFrame(columns=model_columns)

        for col in model_columns:
            val = df.get(col, 0)
            try:
                df_clean.loc[0, col] = float(val)
            except:
                df_clean.loc[0, col] = 0.0

        prediction = model.predict(df_clean)[0]
        
        try:
            prob = model.predict_proba(df_clean)[0][1]
            fraud_prob = round(float(prob) * 100, 2)
        except:
            fraud_prob = "N/A"

        result_text = "FRAUD" if prediction == 1 else "Safe"
        
        return Response({
            'is_fraud': int(prediction),
            'fraud_probability': fraud_prob,
            'message': result_text
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def validation_data(request):
    """
    Reads the REAL creditcard.csv, samples it, and validates the model against it.
    """
    if not os.path.exists(csv_path):
        return Response({'error': 'creditcard.csv file not found on server. Please upload it.'}, status=404)
        
    if model is None:
        return Response({'error': 'Model not loaded'}, status=503)

    try:
        # 1. Load a random sample of the REAL dataset (e.g., 200 rows)
        # Loading the full 200MB file every time is too slow, so we sample.
        df = pd.read_csv(csv_path)
        
        # Try to find the target column (usually 'Class' or 'is_fraud')
        target_col = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
        
        # Take a sample of 200 rows for statistics
        sample_df = df.sample(n=200, random_state=42)
        
        # Prepare inputs for model (drop the answer key)
        X_sample = sample_df.drop(columns=[target_col])
        
        # Ensure columns match what model expects
        X_clean = pd.DataFrame(columns=model_columns)
        for col in model_columns:
             if col in X_sample.columns:
                 X_clean[col] = X_sample[col]
             else:
                 X_clean[col] = 0
        
        # 2. Run REAL Predictions on this sample
        y_pred = model.predict(X_clean)
        y_actual = sample_df[target_col].values

        # 3. Build the Data List for the Frontend Table (First 10 rows)
        table_data = []
        for i in range(10): # Send 10 rows to table
            row_actual = int(y_actual[i])
            row_pred = int(y_pred[i])
            
            table_data.append({
                "time": str(X_clean.iloc[i].get('Time', 'N/A')), # Assuming 'Time' column exists
                "amount": float(X_clean.iloc[i].get('Amount', 0.0)), # Assuming 'Amount' column exists
                "actual": "Fraud" if row_actual == 1 else "Legitimate",
                "predicted": "Fraud" if row_pred == 1 else "Legitimate",
                "match": row_actual == row_pred
            })

        # 4. Calculate Real Statistics for the Charts
        actual_fraud = int(np.sum(y_actual == 1))
        actual_legit = int(np.sum(y_actual == 0))
        pred_fraud = int(np.sum(y_pred == 1))
        pred_legit = int(np.sum(y_pred == 0))
        
        correct_count = int(np.sum(y_actual == y_pred))
        
        # False Alarm: Model said Fraud (1), but it was Safe (0)
        false_alarm = int(np.sum((y_pred == 1) & (y_actual == 0)))
        
        # Missed Fraud: Model said Safe (0), but it was Fraud (1)
        missed_fraud = int(np.sum((y_pred == 0) & (y_actual == 1)))

        stats = {
            "actual_fraud": actual_fraud,
            "actual_legit": actual_legit,
            "pred_fraud": pred_fraud,
            "pred_legit": pred_legit,
            "correct_count": correct_count,
            "false_alarm": false_alarm,
            "missed_fraud": missed_fraud
        }

        return Response({
            'status': 'success',
            'data': table_data,
            'stats': stats,
            'columns': model_columns
        })

    except Exception as e:
        print(f"Error processing real data: {e}")
        return Response({'error': str(e)}, status=500)