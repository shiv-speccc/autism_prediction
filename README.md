
# 🧠 End-to-End Autism Spectrum Disorder Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![License](https://img.shields.io/badge/License-MIT-yellow)


## 📌 Overview

Built an end-to-end Machine Learning pipeline to predict Autism Spectrum Disorder (ASD) using AQ-10 screening questionnaire data.

### Key Highlights
- Performed Exploratory Data Analysis (EDA)
- Handled class imbalance using SMOTE
- Trained and compared Logistic Regression, Random Forest, and XGBoost models
- Evaluated models using Accuracy, ROC-AUC, and Cross Validation
- Automated model persistence and inference pipeline
- Generated visual analytics for model interpretation


## 🛠 Tech Stack

Python • Pandas • NumPy • Scikit-Learn • XGBoost • Matplotlib • Seaborn • Joblib


## 🎯 Why This Project?

Autism diagnosis often requires extensive clinical assessment and specialist evaluation.

This project explores how machine learning can support preliminary ASD screening using AQ-10 behavioural questionnaire data, enabling faster and more scalable risk assessment.


## 🔄 ML Workflow

Dataset
↓
Data Cleaning
↓
EDA
↓
Feature Engineering
↓
SMOTE Balancing
↓
Model Training
↓
Model Evaluation
↓
Best Model Selection
↓
Inference Pipeline


## 📁 Project Structure

```
autism_prediction/
├── data/
│   └── train.csv           ← Dataset (800 samples, AQ-10 questionnaire)
├── models/                 ← Saved model + scaler files (auto-created)
├── outputs/                ← Charts and plots (auto-created)
├── src/
│   ├── train.py            ← Full ML pipeline (EDA → train → evaluate → save)
│   └── predict.py          ← Run inference on a new patient record
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Execution (VS Code — Windows)

### Step 1 — Open the project in VS Code
1. Place the `autism_prediction` folder somewhere convenient (e.g. `C:\Projects\`)
2. Open VS Code → **File → Open Folder** → select `autism_prediction`

---

### Step 2 — Create a virtual environment (Python 3.11 recommended)

Open the **VS Code Terminal** (`Ctrl + `` ` ```) and run:

```bash
# Create venv with Python 3.11
py -3.11 -m venv venv

# Activate it
venv\Scripts\activate
```

> ✅ You should see `(venv)` at the start of the terminal prompt.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn, and joblib.

---

### Step 4 — Train the model

```bash
python src/train.py
```

**What this does:**
- Loads and cleans `data/train.csv`
- Handles class imbalance with SMOTE
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Prints accuracy, AUC-ROC, and cross-validation scores
- Saves model files to `models/`
- Saves 5 charts to `outputs/`

**Expected output:**
```
============================================================
   AUTISM PREDICTION — ML PIPELINE
============================================================
[1] Dataset loaded  →  800 rows × 22 cols
[2] Preprocessing …
[3] Train/Test split  →  Train: 640 | Test: 160
[4] Training models …
    Logistic Regression       Acc=...  AUC=...  CV-AUC=...
    Random Forest             Acc=...  AUC=...  CV-AUC=...
    XGBoost                   Acc=...  AUC=...  CV-AUC=...
[5] Best Model → ...
[6] Saving visualisations …
```

---

### Step 5 — View outputs

Open the `outputs/` folder to find:

| File | Description |
|------|-------------|
| `confusion_matrices.png` | Side-by-side confusion matrices for all 3 models |
| `roc_curves.png` | ROC curves with AUC scores |
| `model_comparison.png` | Bar chart comparing Accuracy vs AUC |
| `feature_importance.png` | Top 15 predictive features (Random Forest) |
| `aq_score_distribution.png` | AQ-10 score distribution by ASD class |

---

### Step 6 — Run inference on a new patient

Edit the `sample` dictionary in `src/predict.py` with a patient's values, then run:

```bash
python src/predict.py
```

**Output example:**
```
=============================================
   AUTISM PREDICTION — INFERENCE
=============================================
  Model used : XGBoost
  Prediction : ⚠  ASD Detected
  Probability: 87.43%
=============================================
```

---


## 🔬 Dataset Features

| Feature | Description |
|---------|-------------|
| A1–A10 Score | AQ-10 behavioural screening questions (0 or 1) |
| age | Patient age |
| gender | Male / Female |
| ethnicity | Ethnic background |
| jaundice | Jaundice at birth (yes/no) |
| austim | Family member with autism (yes/no) |
| contry_of_res | Country of residence |
| used_app_before | Prior screening app usage |
| result | Raw AQ-10 score |
| relation | Who completed the questionnaire |
| **Class/ASD** | **Target: 1 = ASD, 0 = No ASD** |

---

## 🤖 Models Used

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model, uses StandardScaler |
| Random Forest | Ensemble of 200 decision trees |
| XGBoost | Gradient boosting, typically best performer |

**Imbalance handling:** SMOTE (Synthetic Minority Oversampling Technique)  
**Evaluation:** Accuracy, AUC-ROC, Stratified 5-Fold Cross-Validation

---

## ⚠️ Disclaimer
This project is for academic/educational purposes only. It is **not** a clinical diagnostic tool. Always consult a qualified medical professional for autism diagnosis.
