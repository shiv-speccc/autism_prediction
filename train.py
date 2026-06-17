"""
Autism Spectrum Disorder (ASD) Prediction
==========================================
JAIN University | B.Tech CSE Data Science
Author: Shiv | Student ID: 23BTRDC040

Dataset  : AQ-10 Screening Questionnaire (800 samples, 22 features)
Target   : Class/ASD  (0 = No ASD, 1 = ASD)
Models   : Logistic Regression | Random Forest | XGBoost
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for VS Code)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data",    "train.csv")
MODELS = os.path.join(BASE, "models")
OUT    = os.path.join(BASE, "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT,    exist_ok=True)

# ─────────────────────────────────────────────
# 1.  Load & Explore
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("   AUTISM PREDICTION — ML PIPELINE")
print("="*60)

df = pd.read_csv(DATA)
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Target distribution:\n{df['Class/ASD'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 2.  Preprocessing
# ─────────────────────────────────────────────
print("\n[2] Preprocessing …")

# Drop ID — not predictive
df.drop(columns=["ID"], inplace=True)

# Replace '?' placeholders with mode
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace("?", df[col].mode()[0])

# age_desc is redundant (all "18 and more"); drop it
df.drop(columns=["age_desc"], inplace=True)

# Binary encode yes/no & m/f columns
binary_map = {"yes": 1, "no": 0, "m": 1, "f": 0}
for col in ["jaundice", "austim", "used_app_before", "gender"]:
    df[col] = df[col].map(binary_map)

# Label-encode remaining categoricals
le = LabelEncoder()
for col in ["ethnicity", "contry_of_res", "relation"]:
    df[col] = le.fit_transform(df[col].astype(str))

print("    Missing values after cleaning:", df.isnull().sum().sum())
print(f"    Final shape: {df.shape}")

# ─────────────────────────────────────────────
# 3.  Feature / Target Split + SMOTE
# ─────────────────────────────────────────────
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[3] Train/Test split  →  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"    Class imbalance before SMOTE: {y_train.value_counts().to_dict()}")

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"    Class balance after  SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS, "scaler.pkl"))

# ─────────────────────────────────────────────
# 4.  Train Models
# ─────────────────────────────────────────────
print("\n[4] Training models …")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost":              XGBClassifier(n_estimators=200, use_label_encoder=False,
                                          eval_metric="logloss", random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train_sm
    Xte = X_test_sc  if name == "Logistic Regression" else X_test

    model.fit(Xtr, y_train_sm)
    preds   = model.predict(Xte)
    proba   = model.predict_proba(Xte)[:, 1]
    acc     = accuracy_score(y_test, preds)
    auc     = roc_auc_score(y_test, proba)

    cv_scores = cross_val_score(
        model, Xtr, y_train_sm, cv=StratifiedKFold(5), scoring="roc_auc"
    )

    results[name] = {
        "model": model, "preds": preds, "proba": proba,
        "acc": acc, "auc": auc, "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    }

    joblib.dump(model, os.path.join(MODELS, f"{name.replace(' ', '_')}.pkl"))
    print(f"    {name:<25}  Acc={acc:.4f}  AUC={auc:.4f}  CV-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5.  Best Model & Full Report
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]

print(f"\n[5] Best Model → {best_name}  (AUC = {best['auc']:.4f})")
print("\n    Classification Report:")
print(classification_report(y_test, best["preds"],
                            target_names=["No ASD (0)", "ASD (1)"]))

# ─────────────────────────────────────────────
# 6.  Visualisations
# ─────────────────────────────────────────────
print("[6] Saving visualisations …")

# — 6a. Confusion matrices (all models)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No ASD", "ASD"],
                yticklabels=["No ASD", "ASD"])
    ax.set_title(f"{name}\nAcc={r['acc']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=150)
plt.close()

# — 6b. ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curves.png"), dpi=150)
plt.close()

# — 6c. Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names  = list(results.keys())
accs   = [results[n]["acc"]  for n in names]
aucs   = [results[n]["auc"]  for n in names]
x      = np.arange(len(names))
width  = 0.35
ax.bar(x - width/2, accs, width, label="Accuracy", color="#4C72B0")
ax.bar(x + width/2, aucs, width, label="AUC-ROC",  color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0.5, 1.05)
ax.set_title("Model Comparison — Accuracy vs AUC-ROC")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
plt.close()

# — 6d. Feature importance (Random Forest)
rf   = results["Random Forest"]["model"]
feat = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
fig, ax = plt.subplots(figsize=(8, 5))
feat.plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title("Top 15 Feature Importances — Random Forest")
ax.set_ylabel("Importance"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

# — 6e. AQ-10 Score distribution by class
fig, ax = plt.subplots(figsize=(7, 4))
aq_cols = [f"A{i}_Score" for i in range(1, 11)]
df["AQ_Total"] = df[aq_cols].sum(axis=1)
for cls, grp in df.groupby("Class/ASD"):
    ax.hist(grp["AQ_Total"], bins=12, alpha=0.6,
            label=f"Class {cls} ({'ASD' if cls else 'No ASD'})")
ax.set_xlabel("AQ-10 Total Score"); ax.set_ylabel("Count")
ax.set_title("AQ-10 Score Distribution by Class")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "aq_score_distribution.png"), dpi=150)
plt.close()

print("    Saved → outputs/confusion_matrices.png")
print("    Saved → outputs/roc_curves.png")
print("    Saved → outputs/model_comparison.png")
print("    Saved → outputs/feature_importance.png")
print("    Saved → outputs/aq_score_distribution.png")

# ─────────────────────────────────────────────
# 7.  Summary Table
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("   FINAL RESULTS SUMMARY")
print("="*60)
summary = pd.DataFrame({
    "Model":    list(results.keys()),
    "Accuracy": [f"{results[n]['acc']:.4f}" for n in results],
    "AUC-ROC":  [f"{results[n]['auc']:.4f}" for n in results],
    "CV-AUC":   [f"{results[n]['cv_mean']:.4f}±{results[n]['cv_std']:.4f}" for n in results],
})
print(summary.to_string(index=False))
print(f"\n✅ Best Model: {best_name}")
print(f"   Models saved in  → models/")
print(f"   Plots  saved in  → outputs/")
print("="*60 + "\n")
