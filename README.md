VitalGuard — Health Risk Prediction

VitalGuard is a machine learning pipeline that predicts a patient's risk of diabetes from basic vitals (glucose, BMI, age, etc.) and returns both a probability score and a human-readable risk level (Low / Medium / High). The project also includes SHAP-based explainability so predictions can be justified feature-by-feature.

Built on the classic Pima Indians Diabetes Dataset.

Project Pipeline

The project is organized as a 5-step pipeline, meant to be run in order:

Step	Script	Purpose
1	step1_load_data.py	Downloads the Pima Indians Diabetes dataset and saves it as raw_data.csv
2	step2_feature_engineering.py	Cleans invalid zero-values, scales features, and splits into train/test sets
3	step3_train_model.py	Trains an XGBoost and a Random Forest classifier, keeps the better one as model.pkl
4	step4_risk_predictor.py	Exposes a predict_risk() function that turns raw patient vitals into a risk score + level
5	step5_model_report.py	Generates accuracy metrics, a confusion matrix, and SHAP explainability plots
1. Data Loading (step1_load_data.py)

Loads the dataset directly from a public URL (no manual download needed) and adds the standard column names (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome). Saves the result to raw_data.csv.

2. Feature Engineering (step2_feature_engineering.py)
Replaces biologically impossible zero-values (in Glucose, BloodPressure, SkinThickness, Insulin, BMI) with the column median.
Splits data into features (X) and target (y = Outcome).
Scales features with StandardScaler (mean 0, std 1) — the fitted scaler is saved to scaler.pkl so the same transformation can be reused at inference time.
Splits data into 80% train / 20% test (stratified, random_state=42).
Saves X_train.npy, X_test.npy, y_train.npy, y_test.npy for the next step.
3. Model Training (step3_train_model.py)
Trains an XGBoost classifier (100 trees, max depth 4, learning rate 0.1).
Trains a Random Forest classifier (100 trees, max depth 6) as a comparison baseline.
Selects whichever model has the higher test accuracy and saves it to model.pkl.
4. Risk Prediction (step4_risk_predictor.py)

Loads model.pkl and scaler.pkl once, then exposes:

python
predict_risk(
    Pregnancies, Glucose, BloodPressure, SkinThickness,
    Insulin, BMI, DiabetesPedigreeFunction, Age
) -> {"risk_probability": float, "risk_level": "Low" | "Medium" | "High"}

Risk levels are derived from the predicted probability:

Low: probability < 0.40
Medium: 0.40 ≤ probability < 0.70
High: probability ≥ 0.70

This module is intended to be imported directly by a backend service (e.g. a FastAPI endpoint) to serve real-time predictions.

5. Model Report & Explainability (step5_model_report.py)
Computes accuracy, F1 score, and a full classification report on the test set.
Renders and saves a confusion matrix (confusion_matrix.png).
Uses SHAP (TreeExplainer) to compute per-feature contributions to each prediction, saving:
shap_summary.png — a bar chart of overall feature importance
shap_values.npy — raw SHAP values, for use in a downstream explanation UI
Current Results

Based on the included confusion_matrix.png:

	Predicted: No Diabetes	Predicted: Diabetes
Actual: No Diabetes	86	14
Actual: Diabetes	25	29
Accuracy: ~74.7% (115/154 correct on the test set)
F1 score (Diabetes class): ~0.60

The SHAP summary plot (shap_summary.png) highlights Glucose and Pregnancies as the features with the largest influence on predicted risk.

Project Files
File	Description
step1_load_data.py – step5_model_report.py	Pipeline scripts (run in order)
raw_data.csv	Raw dataset saved by Step 1
X_train.npy, X_test.npy, y_train.npy, y_test.npy	Scaled train/test splits saved by Step 2
scaler.pkl	Fitted StandardScaler, required for consistent inference-time scaling
model.pkl	Trained classifier (best of XGBoost / Random Forest)
confusion_matrix.png	Visual summary of model performance on the test set
shap_summary.png	Bar chart of feature importance via SHAP
shap_values.npy	Raw SHAP values for the test set
How to Run
bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib

python step1_load_data.py
python step2_feature_engineering.py
python step3_train_model.py
python step4_risk_predictor.py   # optional: runs a sample prediction
python step5_model_report.py

Each script reads the output files of the previous step, so they must be run in order from the same working directory.

Using the Predictor in Another Application
python
from step4_risk_predictor import predict_risk

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

print(result)
# {"risk_probability": 0.xx, "risk_level": "Medium" | "High"}

model.pkl and scaler.pkl must be present in the working directory (or their paths updated in step4_risk_predictor.py) for this to work.

Notes
The dataset uses 0 as a placeholder for missing values in several medical columns; these are imputed with the column median during feature engineering rather than being treated as real zero readings.
The scaler used at training time must always be reused at prediction time — this is why scaler.pkl is saved and loaded alongside model.pkl.
SHAP explainability output (shap_values.npy, shap_summary.png) is intended to support a front-end explanation feature (e.g. "why was this patient flagged as high risk?").
