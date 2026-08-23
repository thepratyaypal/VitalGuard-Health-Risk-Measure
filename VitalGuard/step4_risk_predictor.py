# ─────────────────────────────────────────────────────────
# STEP 4: Risk Score Logic
# Member 2 (Backend) will import predict_risk() from here
# ─────────────────────────────────────────────────────────

import numpy as np
import joblib

# Load model and scaler once when this file is imported
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level label."""
    if probability < 0.40:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    else:
        return "High"


def predict_risk(
    Pregnancies: float,
    Glucose: float,
    BloodPressure: float,
    SkinThickness: float,
    Insulin: float,
    BMI: float,
    DiabetesPedigreeFunction: float,
    Age: float
) -> dict:
    """
    Takes patient vitals and returns diabetes risk.

    Returns a dict like:
    {
        "risk_probability": 0.73,
        "risk_level": "High"
    }
    """
    # 1. Pack inputs into a 2D array (model expects this shape)
    input_data = np.array([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]])

    # 2. Scale using the SAME scaler from training
    input_scaled = scaler.transform(input_data)

    # 3. Predict probability of diabetes (column index 1 = positive class)
    probability = float(model.predict_proba(input_scaled)[0][1])

    # 4. Get risk level
    risk_level = get_risk_level(probability)

    return {
        "risk_probability": round(probability, 4),
        "risk_level": risk_level
    }


# ── TEST IT ─────────────────────────────────────────────
# Run this file directly to test
if __name__ == "__main__":
    # Sample patient data
    result = predict_risk(
        Pregnancies=6,
        Glucose=148,
        BloodPressure=72,
        SkinThickness=35,
        Insulin=0,
        BMI=33.6,
        DiabetesPedigreeFunction=0.627,
        Age=50
    )
    print("Risk Probability:", result["risk_probability"])
    print("Risk Level:",       result["risk_level"])
    # Expected output: Medium or High risk