import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- 1. SETUP ---
# Ensure the CSV file path is correct
csv_path = 'creditcard.csv' 
if not os.path.exists(csv_path):
    print("FATAL: creditcard.csv not found. Please ensure it's in the current folder.")
    exit()

print("Loading and preprocessing data...")
df = pd.read_csv(csv_path)

# 2. PREPARE DATA (Matches previous successful run)
# We use the full dataset here (X, y) for model training and full stats calculation
target_col = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
y_full = df[target_col]
X_full = df.drop(columns=[target_col])
X_full = X_full.select_dtypes(include=[np.number])
X_full = X_full.fillna(0)

# Check the total size of the full dataset
total_rows = len(X_full)
print(f"Total rows loaded: {total_rows}")

# 3. TRAIN & TEST MODEL (Standard practice, using train/test splits)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. GENERATE STATS ON THE FULL DATASET (To display 250k on the dashboard) ---
# We use the trained model to predict on the ENTIRE dataset (X_full)
# This provides the large numbers needed for the dashboard's "Accuracy on all data" visualization.

# Calculate final predictions on the full dataset
y_pred_full = model.predict(X_full)
cm_full = confusion_matrix(y_full, y_pred_full)

# Confusion Matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm_full.ravel()

# Calculate statistics needed for your dashboard charts
stats_data = {
    # 'actual' counts should be identical to the full dataset class distribution
    "actual_fraud": int(y_full.sum()),
    "actual_legit": int(total_rows - y_full.sum()),
    
    "pred_fraud": int(TP + FP),       # Total predicted as fraud
    "pred_legit": int(TN + FN),       # Total predicted as legit
    
    "correct_count": int(TN + TP),    # Total correct (Accuracy)
    "false_alarm": int(FP),           # False Positives (Legit flagged as Fraud)
    "missed_fraud": int(FN),          # False Negatives (Fraud missed as Legit)
    "total_transactions": total_rows  # ADDED: Explicit total count
}

# 5. SAVE ARTIFACTS
joblib.dump(model, 'fraud_model_pipeline.pkl')
joblib.dump(X_full.columns.tolist(), 'model_columns.pkl')

# SAVE THE ACCURATE STATS
with open('model_stats.json', 'w') as f:
    json.dump(stats_data, f, indent=4)

print("✅ All files saved. model_stats.json now reflects the full dataset size.")
print(f"Stats Data Summary:\n{json.dumps(stats_data, indent=4)}")

# The total of the 'actual' counts should now be equal to total_rows (250,000 or whatever your CSV size is).