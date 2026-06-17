"""
predict.py — Run inference on a new patient record.
Usage:
    python src/predict.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

# ── Load artefacts ──────────────────────────────────────────────
best_model_name = "XGBoost"          # change if a different model won
model  = joblib.load(os.path.join(MODELS, f"{best_model_name.replace(' ', '_')}.pkl"))
scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))

FEATURES = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "ethnicity", "jaundice", "austim",
    "contry_of_res", "used_app_before", "result", "relation",
]

# ── Sample patient (edit these values) ─────────────────────────
sample = {
    "A1_Score":        1,          # 0 or 1
    "A2_Score":        0,
    "A3_Score":        1,
    "A4_Score":        1,
    "A5_Score":        0,
    "A6_Score":        1,
    "A7_Score":        1,
    "A8_Score":        1,
    "A9_Score":        0,
    "A10_Score":       1,
    "age":             25,
    "gender":          1,          # 1=male, 0=female
    "ethnicity":       3,          # label-encoded integer (see train.py)
    "jaundice":        0,          # 1=yes, 0=no
    "austim":          0,          # family member with autism? 1=yes, 0=no
    "contry_of_res":   10,         # label-encoded integer
    "used_app_before": 0,          # 1=yes, 0=no
    "result":          7.5,        # AQ-10 screening result score
    "relation":        4,          # label-encoded integer
}

X_new = pd.DataFrame([sample])

# XGBoost doesn't need scaling; LR does — adjust if needed
pred  = model.predict(X_new)[0]
prob  = model.predict_proba(X_new)[0][1]

print("\n" + "="*45)
print("   AUTISM PREDICTION — INFERENCE")
print("="*45)
print(f"  Model used : {best_model_name}")
print(f"  Prediction : {'⚠  ASD Detected' if pred == 1 else '✅  No ASD Detected'}")
print(f"  Probability: {prob:.2%}")
print("="*45 + "\n")
print("Note: This is a screening tool only.")
print("Always consult a qualified medical professional.\n")
