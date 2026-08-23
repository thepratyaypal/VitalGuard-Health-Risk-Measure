# ─────────────────────────────────────────────────────────
# STEP 5: Model Accuracy Report + SHAP Explainability
# ─────────────────────────────────────────────────────────

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ── LOAD EVERYTHING ─────────────────────────────────────
model  = joblib.load("model.pkl")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ── PREDICTIONS ─────────────────────────────────────────
y_pred = model.predict(X_test)

# ── PRINT METRICS ───────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print("=" * 40)
print("  VitalGuard — Model Report")
print("=" * 40)
print(f"  Accuracy : {acc * 100:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# ── CONFUSION MATRIX CHART ──────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)
plt.title("Confusion Matrix — VitalGuard")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("✅ Saved: confusion_matrix.png")

# ── SHAP EXPLAINABILITY ─────────────────────────────────
# SHAP tells us: which features push the prediction up or down?
# This is what Member 3 uses for the explanation UI

# Create SHAP explainer — TreeExplainer works great for XGBoost/RF
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle both XGBoost (2D) and Random Forest (list of 2) output
if isinstance(shap_values, list):
    sv = shap_values[1]   # RF: index 1 = diabetes class
else:
    sv = shap_values       # XGBoost: single array

# SHAP Summary Plot (bar chart of feature importance)
plt.figure(figsize=(8, 5))
shap.summary_plot(
    sv, X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance — VitalGuard")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: shap_summary.png")

# Save SHAP values as numpy file for Member 3 (frontend explainability)
np.save("shap_values.npy", sv)
print("✅ Saved: shap_values.npy")

print()
print("🎉 All reports generated successfully!")