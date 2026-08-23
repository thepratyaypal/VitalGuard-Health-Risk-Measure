# ─────────────────────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the raw data we saved in Step 1
df = pd.read_csv("raw_data.csv")

# ── FIX MISSING VALUES ──────────────────────────────────
# These columns cannot be 0 in real life → replace with median
zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_not_allowed:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)

print("✅ Missing values fixed (0s replaced with median)")

# ── SEPARATE FEATURES AND LABEL ─────────────────────────
# X = input features (what the doctor enters)
# y = target label (0 = no diabetes, 1 = diabetes)
X = df.drop("Outcome", axis=1)   # All columns except Outcome
y = df["Outcome"]                  # Only the Outcome column

# ── SCALE THE DATA ──────────────────────────────────────
# StandardScaler makes all features have mean=0, std=1
# This helps the model treat all features equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on data, then transform

# ── SPLIT DATA ──────────────────────────────────────────
# 80% training, 20% testing
# random_state=42 means results are reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── SAVE SCALER (VERY IMPORTANT!) ───────────────────────
# The backend must use THIS SAME scaler when making predictions
# If you don't save it, new inputs won't be scaled correctly!
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved to scaler.pkl")

# Save processed data for Step 3
np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)
print("✅ Processed data saved")