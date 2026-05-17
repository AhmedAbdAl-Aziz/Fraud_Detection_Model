"""
train.py — LightGBM Fraud Detection Model Training Pipeline
============================================================

الاستخدام:
  python generate_data.py   # أولاً: ولّد البيانات
  python train.py           # ثانياً: درّب الموديل

الملفات الناتجة:
  lgbm_fraud_model.pkl   — الموديل المدرَّب
  model_artifacts.pkl    — encoders + feature list + thresholds
  sender_profiles.pkl    — behavioral profiles للـ inference
"""

import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

CSV_PATH       = "synthetic_transactions.csv"
MODEL_PATH     = "lgbm_fraud_model.pkl"
ARTIFACTS_PATH = "model_artifacts.pkl"
PROFILES_PATH  = "sender_profiles.pkl"

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SecureB2B — LightGBM Training Pipeline")
print("=" * 60)

print("\n[1/7] Loading dataset …")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"      Rows: {len(df):,}  |  Fraud: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")

# ── 2. Build Sender Behavioral Profiles ───────────────────────────────────────
print("\n[2/7] Building behavioral profiles …")
profiles = (
    df.groupby("sender_company_name")
    .agg(
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        transaction_count=("amount", "count"),
    )
    .reset_index()
)
profiles["std_amount"] = profiles["std_amount"].fillna(0)
profiles.to_pickle(PROFILES_PATH)
print(f"      Profiles for {len(profiles)} companies → '{PROFILES_PATH}'")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("\n[3/7] Engineering features …")

# Merge sender profile stats (the CSV already has avg/std but we recompute from data)
df = df.merge(
    profiles.rename(columns={"avg_amount": "calc_avg", "std_amount": "calc_std"}),
    on="sender_company_name",
    how="left",
)

# Z-score: deviation from sender's historical average
df["z_score"] = np.where(
    df["calc_std"] > 0,
    (df["amount"] - df["calc_avg"]) / df["calc_std"],
    0.0,
).clip(-10, 10)

# Amount-to-average ratio
df["amount_to_avg_ratio"] = np.where(
    df["calc_avg"] > 0,
    (df["amount"] / df["calc_avg"]).clip(0, 50),
    1.0,
)

# Log transforms
df["log_amount"]        = np.log1p(df["amount"])
df["log_avg_amount"]    = np.log1p(df["calc_avg"])
df["log_tx_count"]      = np.log1p(df["transaction_count"])
df["log_sender_age"]    = np.log1p(df["sender_age_days"])

# Business hours flag
df["is_business_hours"] = df["hour_of_day"].between(8, 18).astype(int)
df["is_night"]          = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)

# Label encode categorical columns
sender_encoder   = LabelEncoder()
receiver_encoder = LabelEncoder()
industry_encoder = LabelEncoder()

df["sender_encoded"]   = sender_encoder.fit_transform(df["sender_company_name"].fillna("UNKNOWN"))
df["receiver_encoded"] = receiver_encoder.fit_transform(df["receiver_company_name"].fillna("UNKNOWN"))
df["industry_encoded"] = industry_encoder.fit_transform(df["industry"].fillna("UNKNOWN"))

FEATURES = [
    # Core amount features
    "amount",
    "log_amount",
    "log_avg_amount",
    "amount_to_avg_ratio",

    # Behavioral deviation
    "z_score",
    "calc_avg",
    "calc_std",
    "log_tx_count",

    # Risk signal from OCR
    "risk_score",

    # Sender profile
    "log_sender_age",
    "sender_avg_amount",
    "sender_std_amount",

    # Time features
    "hour_of_day",
    "is_weekend",
    "is_business_hours",
    "is_night",

    # Categorical (encoded)
    "sender_encoded",
    "receiver_encoded",
    "industry_encoded",
]

X = df[FEATURES].fillna(0)
y = df["label"]

print(f"      Features: {len(FEATURES)}")
print(f"      X shape : {X.shape}")

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
print("\n[4/7] Splitting data …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. Train LightGBM ─────────────────────────────────────────────────────────
print("\n[5/7] Training LightGBM …")

params = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "num_leaves":        127,
    "max_depth":         -1,
    "learning_rate":     0.03,
    "n_estimators":      1000,
    "min_child_samples": 15,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.05,
    "reg_lambda":        0.1,
    "random_state":      42,
    "verbose":           -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(200),
    ],
)

print(f"      Best iteration: {model.best_iteration_}")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n[6/7] Evaluating …")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
print(f"\n  ✅ ROC-AUC Score: {auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print("  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
fi = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\n  Top 10 Feature Importances:")
print(fi.head(10).to_string(index=False))

# ── 7. Save Model & Artifacts ─────────────────────────────────────────────────
print("\n[7/7] Saving model …")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

artifacts = {
    "sender_encoder":   sender_encoder,
    "receiver_encoder": receiver_encoder,
    "industry_encoder": industry_encoder,
    "features":         FEATURES,
    "thresholds": {
        "reject":  0.65,   # prob >= 0.65 → AUTO_REJECT
        "approve": 0.30,   # prob <= 0.30 → AUTO_APPROVE
    },
    "auc_score": auc,
    "model_version": "lgbm_v1",
}

with open(ARTIFACTS_PATH, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n  ✅ Model     → '{MODEL_PATH}'")
print(f"  ✅ Artifacts → '{ARTIFACTS_PATH}'")
print(f"  ✅ Profiles  → '{PROFILES_PATH}'")
print(f"\n  ROC-AUC: {auc:.4f}")
print("\nUpload the 3 .pkl files to GitHub and redeploy on Railway. 🚀")
