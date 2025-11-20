import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- 1. SETUP ---
csv_path = 'creditcard.csv'
if not os.path.exists(csv_path):
    print("FATAL: creditcard.csv not found. Please ensure it's in the current folder.")
    exit()

print("Loading and preprocessing data...")
df = pd.read_csv(csv_path)

# 2. PREPARE DATA (Matches previous successful run)
target_col = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
y = df[target_col]
X = df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

# 3. TRAIN & TEST MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 4. GENERATE REAL STATS ON THE TEST SET (100% Accuracy Data)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()

# Calculate statistics needed for your dashboard charts
stats_data = {
    "actual_fraud": int(TP + FN),
    "actual_legit": int(TN + FP),
    "pred_fraud": int(TP + FP), # Total predicted as fraud
    "pred_legit": int(TN + FN), # Total predicted as legit
    "correct_count": int(TN + TP), # Total correct
    "false_alarm": int(FP),        # False Positives (Safe but flagged)
    "missed_fraud": int(FN)       # False Negatives (Fraud but missed)
}

# 5. SAVE ARTIFACTS
joblib.dump(model, 'fraud_model_pipeline.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# SAVE THE ACCURATE STATS
with open('model_stats.json', 'w') as f:
    json.dump(stats_data, f, indent=4)

print("✅ All files saved.")