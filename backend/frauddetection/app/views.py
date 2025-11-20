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
# We wrap this in try-except so the server starts even if files are missing.
model = None
model_columns = []

try:
    print("--- 🛡️ SYSTEM STARTUP: Loading Model ---")
    
    # Construct paths safely
    base_path = settings.BASE_DIR
    model_path = os.path.join(base_path, 'fraud_model_pipeline.pkl')
    columns_path = os.path.join(base_path, 'model_columns.pkl')

    # Check if files exist before trying to load
    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        print("✅ SUCCESS: Model loaded. System is ready to predict.")
    else:
        print("⚠️ WARNING: Model files not found at:")
        print(f"   - {model_path}")
        print("   The server is running, but predictions will return errors.")

except Exception as e:
    # If loading fails for ANY reason (bad file, wrong version, etc.), catch it here.
    print(f"❌ CRITICAL ERROR during startup: {str(e)}")
    print("   The server is running in Safe Mode (No ML).")


# ==========================================
# 2. SAFE PREDICTION ENDPOINT
# ==========================================
@api_view(['POST'])
def predict(request):
    """
    This function handles requests safely. 
    If anything fails, it returns a clean error message, NOT a crash.
    """
    
    # --- Safety Check 1: Is the model loaded? ---
    if model is None:
        return Response(
            {'error': 'Service Unavailable: The fraud detection model is not active.'}, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        # --- Safety Check 2: Get Data ---
        input_data = request.data
        if not input_data:
            return Response({'error': 'No data provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # --- Safety Check 3: Process Data ---
        # Create DataFrame from input
        df = pd.DataFrame([input_data])

        # Create a clean DataFrame with exactly the columns the model needs
        df_clean = pd.DataFrame(columns=model_columns)

        # Map the data safely
        for col in model_columns:
            # If the user sent this column, use it. If not, use 0.
            val = df.get(col, 0)
            
            # Ensure the value is a number (clean out text/strings)
            try:
                df_clean.loc[0, col] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                df_clean.loc[0, col] = 0.0  # If text was sent instead of a number, make it 0

        # --- Safety Check 4: Predict ---
        prediction = model.predict(df_clean)[0]
        
        # Try to get probability, but don't crash if the model doesn't support it
        try:
            prob = model.predict_proba(df_clean)[0][1]
            fraud_probability = round(float(prob) * 100, 2)
        except:
            fraud_probability = "N/A"

        # Return Result
        result_text = "FRAUD" if prediction == 1 else "Safe"
        
        return Response({
            'is_fraud': int(prediction),
            'fraud_probability': fraud_probability,
            'status': 'success',
            'message': result_text
        })

    except Exception as e:
        # --- FINAL SAFETY NET ---
        # If ANYTHING else goes wrong (pandas error, math error, etc.)
        # Print the error to your terminal so you can fix it
        print(f"❌ RUNTIME ERROR: {str(e)}")
        
        # Return a polite error to the user
        return Response(
            {'error': 'An internal error occurred while processing the transaction.'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )