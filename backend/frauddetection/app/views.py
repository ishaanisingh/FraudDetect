import joblib
import pandas as pd
import os
import numpy as np
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# ==========================================
# 1. SAFE MODEL LOADING (Startup)
# ==========================================
model = None
model_columns = []

try:
    print("--- 🛡️ SYSTEM STARTUP: Loading Model ---")
    
    base_path = settings.BASE_DIR
    model_path = os.path.join(base_path, 'fraud_model_pipeline.pkl')
    columns_path = os.path.join(base_path, 'model_columns.pkl')

    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        print("✅ SUCCESS: Model loaded. System is ready to predict.")
    else:
        print("⚠️ WARNING: Model files not found.")

except Exception as e:
    print(f"❌ CRITICAL ERROR during startup: {str(e)}")


# ==========================================
# 2. PREDICTION ENDPOINT
# ==========================================
@api_view(['POST'])
def predict(request):
    if model is None:
        return Response({'error': 'Service Unavailable: Model not active.'}, status=503)

    try:
        input_data = request.data
        if not input_data:
            return Response({'error': 'No data provided.'}, status=400)

        df = pd.DataFrame([input_data])
        df_clean = pd.DataFrame(columns=model_columns)

        for col in model_columns:
            val = df.get(col, 0)
            try:
                df_clean.loc[0, col] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                df_clean.loc[0, col] = 0.0

        prediction = model.predict(df_clean)[0]
        
        try:
            prob = model.predict_proba(df_clean)[0][1]
            fraud_probability = round(float(prob) * 100, 2)
        except:
            fraud_probability = "N/A"

        result_text = "FRAUD" if prediction == 1 else "Safe"
        
        return Response({
            'is_fraud': int(prediction),
            'fraud_probability': fraud_probability,
            'status': 'success',
            'message': result_text
        })

    except Exception as e:
        print(f"❌ RUNTIME ERROR: {str(e)}")
        return Response({'error': 'An internal error occurred.'}, status=500)


# ==========================================
# 3. VALIDATION DATA ENDPOINT (New)
# ==========================================
@api_view(['GET'])
def validation_data(request):
    if not model_columns:
        return Response({'error': 'Model columns not loaded'}, status=500)
    
    return Response({
        'status': 'success',
        'columns': model_columns
    })