# ─────────────────────────────────────────────────────────
# STEP 3: Train Model & Save as .pkl
# ─────────────────────────────────────────────────────────

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load processed data from Step 2
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

# ── TRAIN XGBOOST ───────────────────────────────────────
xgb_model = XGBClassifier(
    n_estimators=100,       # 100 decision trees
    max_depth=4,            # How deep each tree goes
    learning_rate=0.1,      # How fast model learns
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost Accuracy:      {xgb_acc:.4f}")

# ── TRAIN RANDOM FOREST ─────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ── PICK THE BETTER MODEL ───────────────────────────────
if xgb_acc >= rf_acc:
    best_model = xgb_model
    print("✅ Best Model: XGBoost")
else:
    best_model = rf_model
    print("✅ Best Model: Random Forest")

# ── SAVE THE MODEL ──────────────────────────────────────
# This is the file Member 2 (backend) will load in FastAPI!
joblib.dump(best_model, "model.pkl")
print("✅ Best model saved to model.pkl")

# ── QUICK TEST: Load model and predict ──────────────────
# This proves the .pkl works correctly
loaded_model = joblib.load("model.pkl")
test_pred = loaded_model.predict_proba(X_test[:1])
print(f"Test prediction (probability): {test_pred[0][1]:.4f}")
print("✅ model.pkl loads and predicts correctly!")