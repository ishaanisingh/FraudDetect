import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 1. Load your dataset
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# 2. Clean the data
# We separate the Target (y) from the Features (X)
y = df['is_fraud']  # <--- UPDATED to your actual column name

# Drop the target from X so the model doesn't cheat
X = df.drop('is_fraud', axis=1)

# IMPORTANT: Keep only numeric columns (ML cannot read 'Names' or 'Dates' directly)
# This prevents "ValueError: could not convert string to float"
X = X.select_dtypes(include=[np.number])

# Fill any missing numbers with 0 (just in case)
X = X.fillna(0)

print(f"Training on {len(X.columns)} numeric features...")

# 3. Train the model
print("Training model... (This might take a minute)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# 4. Save the "Brain" to a file
joblib.dump(model, 'fraud_model_pipeline.pkl')

# 5. ALSO SAVE the column names! 
# (We need this later to know which inputs the website should expect)
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("✅ Success! Model saved as 'fraud_model_pipeline.pkl'")
print("✅ Column names saved as 'model_columns.pkl'")