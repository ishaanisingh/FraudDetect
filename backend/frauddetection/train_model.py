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

# 2. PREPARE DATA 
target_col = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
y_full = df[target_col]
X_full = df.drop(columns=[target_col])
X_full = X_full.select_dtypes(include=[np.number])
X_full = X_full.fillna(0)

total_rows = len(X_full)
print(f"Total rows loaded: {total_rows}")

# 3. TRAIN & TEST MODEL (CRITICAL CHANGE HERE)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# --- CALCULATE CLASS WEIGHTS TO HANDLE IMBALANCE ---
# This weight biases the model to pay more attention to the small fraud class (y=1).
positive_class_count = y_train.sum()
negative_class_count = len(y_train) - positive_class_count
scale_weight = negative_class_count / positive_class_count

# Re-initialize the model with the class weights
model = lgb.LGBMClassifier(
    random_state=42,
    scale_pos_weight=scale_weight, # <--- THIS IS THE FIX for high missed fraud
    n_estimators=100,
    learning_rate=0.05
)

model.fit(X_train, y_train)
print("Model training complete with class weights.")

# --- 4. GENERATE STATS ON THE FULL DATASET (To display 250k on the dashboard) ---
# We use the trained model to predict on the ENTIRE dataset (X_full)
y_pred_full = model.predict(X_full)
cm_full = confusion_matrix(y_full, y_pred_full)

# Confusion Matrix: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm_full.ravel()

# Calculate statistics needed for your dashboard charts
stats_data = {
    "actual_fraud": int(y_full.sum()),
    "actual_legit": int(total_rows - y_full.sum()),
    
    "pred_fraud": int(TP + FP),       
    "pred_legit": int(TN + FN),       
    
    "correct_count": int(TN + TP),    
    "false_alarm": int(FP),           
    "missed_fraud": int(FN),          # This value should be much lower now!
    "total_transactions": total_rows  
}

# 5. SAVE ARTIFACTS
joblib.dump(model, 'fraud_model_pipeline.pkl')
joblib.dump(X_full.columns.tolist(), 'model_columns.pkl')

# SAVE THE ACCURATE STATS
with open('model_stats.json', 'w') as f:
    json.dump(stats_data, f, indent=4)

print("✅ All files saved. New model trained with class weights.")
print(f"Stats Data Summary:\n{json.dumps(stats_data, indent=4)}")