# ─────────────────────────────────────────────────────────
# STEP 1: Install Libraries & Load Dataset
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

# Load dataset directly from URL — no download needed!
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# The dataset has no header, so we add column names manually
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

# Quick look at the data
print("Shape:", df.shape)        # Should be (768, 9)
print(df.head())               # First 5 rows
print(df.describe())           # Stats summary
print(df["Outcome"].value_counts())  # 0 = No diabetes, 1 = Diabetes

# Save for next step
df.to_csv("raw_data.csv", index=False)
print("✅ Data loaded and saved to raw_data.csv")