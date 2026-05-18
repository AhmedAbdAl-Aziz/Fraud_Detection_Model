"""
main.py — SecureB2B Fraud Detection Engine v3.0 (LightGBM)
============================================================

Primary classifier: LightGBM (trained on synthetic_transactions.csv)
Fallback:           Z-Score behavioral analysis
"""

import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="SecureB2B Fraud Detection Engine",
    version="3.0.0 — LightGBM",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
lgbm_model       = None   # LightGBM classifier
model_artifacts  = None   # encoders + feature list + thresholds
active_profiles  = None   # behavioral profiles (Z-score fallback)

PROFILES_PATH  = os.environ.get("PROFILES_PATH",  "sender_profiles.pkl")
MODEL_PATH     = os.environ.get("MODEL_PATH",     "lgbm_fraud_model.pkl")
ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "model_artifacts.pkl")
CSV_PATH       = os.environ.get("CSV_PATH",       "synthetic_transactions.csv")


# ---------------------------------------------------------------------------
# Startup: load all artifacts
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_artifacts():
    global lgbm_model, model_artifacts, active_profiles

    # ── 1. LightGBM model ──────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH) and os.path.exists(ARTIFACTS_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                lgbm_model = pickle.load(f)
            with open(ARTIFACTS_PATH, "rb") as f:
                model_artifacts = pickle.load(f)
            auc = model_artifacts.get("auc_score", "N/A")
            print(f"✅ LightGBM model loaded (AUC={auc:.4f})")
        except Exception as e:
            print(f"⚠️  Failed to load LightGBM model: {e}")
            lgbm_model = None
    else:
        print("⚠️  LightGBM model not found — using Z-Score fallback.")

    # ── 2. Behavioral profiles (Z-Score fallback) ──────────────────────────
    if os.path.exists(PROFILES_PATH):
        try:
            active_profiles = pd.read_pickle(PROFILES_PATH)
            print(f"✅ Behavioral profiles loaded ({len(active_profiles)} companies)")
            return
        except Exception as e:
            print(f"⚠️  Failed to load profiles: {e}")

    # Auto-build from CSV if profiles not found
    if os.path.exists(CSV_PATH):
        print(f"🔨 Building profiles from '{CSV_PATH}' …")
        try:
            df = pd.read_csv(CSV_PATH, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()

            amount_col = "amount"
            if amount_col in df.columns:
                df[amount_col] = (
                    df[amount_col].astype(str)
                    .str.replace(" USD", "", regex=False)
                    .str.replace(",", "", regex=False)
                )
                df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

            name_col = "sender_company_name"
            if name_col in df.columns and amount_col in df.columns:
                active_profiles = (
                    df.groupby(name_col)
                    .agg(avg_amount=(amount_col, "mean"),
                         std_amount=(amount_col, "std"),
                         transaction_count=(amount_col, "count"))
                    .reset_index()
                )
                active_profiles["std_amount"] = active_profiles["std_amount"].fillna(0)
                active_profiles.to_pickle(PROFILES_PATH)
                print(f"✅ Profiles built ({len(active_profiles)} companies)")
                return
        except Exception as e:
            print(f"❌ Failed to build profiles from CSV: {e}")

    # Final fallback
    print("⚠️  No profiles available — Z-Score disabled.")
    active_profiles = pd.DataFrame(
        columns=["sender_company_name", "avg_amount", "std_amount", "transaction_count"]
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    Id: str
    sender_company_name: str
    receiver_company_name: str
    amount: float
    currency: str = "USD"
    risk_score: float                        # OCR risk score (0-100)
    average_amount: Optional[float] = None  # Hint from company profile API
    industry: Optional[str] = None          # Sender's industry
    sender_age_days: Optional[int] = None   # How old is the sending company
    hour_of_day: Optional[int] = None       # Hour of transaction (0-23)
    is_business_hours: Optional[int] = None # 1 if within working hours, 0 otherwise
    is_weekend: Optional[int] = None        # 1 if weekend, 0 otherwise
    is_night: Optional[int] = None          # 1 if night (23-5), 0 otherwise


class TransactionDecision(BaseModel):
    transaction_id: str
    final_risk_score: float
    fraud_probability: Optional[float]
    decision: str
    reasons: List[str]
    engine: str                              # "LightGBM" or "Z-Score"


# ---------------------------------------------------------------------------
# LightGBM inference
# ---------------------------------------------------------------------------

def _safe_encode(encoder, value: str, fallback: int = 0) -> int:
    """Encode a label safely — returns fallback index for unseen labels."""
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return fallback


def predict_with_lgbm(tx: TransactionInput) -> dict:
    """
    Build the feature vector and run LightGBM inference.
    Returns {'fraud_prob': float, 'decision': str, 'reasons': list}
    """
    arts     = model_artifacts
    encoders = arts

    # ── Behavioral stats from profiles ──────────────────────────────────────
    profile_row = None
    if active_profiles is not None and not active_profiles.empty:
        mask = active_profiles["sender_company_name"] == tx.sender_company_name
        if mask.any():
            profile_row = active_profiles[mask].iloc[0]

    avg_amount = float(profile_row["avg_amount"]) if profile_row is not None else (tx.average_amount or tx.amount)
    std_amount = float(profile_row["std_amount"]) if profile_row is not None else (avg_amount * 0.3)
    tx_count   = float(profile_row["transaction_count"]) if profile_row is not None else 10.0

    # ── Feature engineering (mirrors train.py) ─────────────────────────────
    z_score = (tx.amount - avg_amount) / std_amount if std_amount > 0 else 0.0
    z_score = float(np.clip(z_score, -10, 10))

    amount_to_avg_ratio = float(np.clip(tx.amount / avg_amount, 0, 50)) if avg_amount > 0 else 1.0

    # Use provided values from Laravel, or fallback to neutral
    hour           = tx.hour_of_day if tx.hour_of_day is not None else 12
    is_weekend     = tx.is_weekend if tx.is_weekend is not None else 0
    is_biz_hours   = tx.is_business_hours if tx.is_business_hours is not None else 1
    is_night       = tx.is_night if tx.is_night is not None else 0

    sender_age   = float(tx.sender_age_days or 365)
    industry_val = tx.industry or "Unknown"

    features = {
        "amount":               tx.amount,
        "log_amount":           float(np.log1p(tx.amount)),
        "log_avg_amount":       float(np.log1p(avg_amount)),
        "amount_to_avg_ratio":  amount_to_avg_ratio,
        "z_score":              z_score,
        "calc_avg":             avg_amount,
        "calc_std":             std_amount,
        "log_tx_count":         float(np.log1p(tx_count)),
        "risk_score":           tx.risk_score,
        "log_sender_age":       float(np.log1p(sender_age)),
        "sender_avg_amount":    tx.average_amount or avg_amount,
        "sender_std_amount":    std_amount,
        "hour_of_day":          hour,
        "is_weekend":           is_weekend,
        "is_business_hours":    is_biz_hours,
        "is_night":             is_night,
        "sender_encoded":       _safe_encode(arts["sender_encoder"],   tx.sender_company_name),
        "receiver_encoded":     _safe_encode(arts["receiver_encoder"], tx.receiver_company_name),
        "industry_encoded":     _safe_encode(arts["industry_encoder"], industry_val),
    }

    feature_order = arts["features"]
    X = np.array([[features[f] for f in feature_order]], dtype=float)

    fraud_prob = float(lgbm_model.predict_proba(X)[0, 1])

    # ── Decision ─────────────────────────────────────────────────────────────
    thresholds = arts["thresholds"]
    reject_th  = thresholds.get("reject",  0.65)
    approve_th = thresholds.get("approve", 0.30)

    if fraud_prob >= reject_th:
        decision = "AUTO_REJECT"
    elif fraud_prob <= approve_th:
        decision = "AUTO_APPROVE"
    else:
        decision = "MANUAL_REVIEW"

    # ── Reasons ───────────────────────────────────────────────────────────────
    reasons = []
    if z_score > 2.5:
        reasons.append(f"Amount (${tx.amount:,.2f}) is {z_score:.1f}σ above sender's historical average.")
    if amount_to_avg_ratio > 3.0:
        reasons.append(f"Transaction is {amount_to_avg_ratio:.1f}x the sender's typical amount.")
    if tx.risk_score > 65:
        reasons.append(f"High OCR risk score ({tx.risk_score:.0f}/100) from invoice analysis.")
    if sender_age < 30:
        reasons.append("Sender is a newly registered company (< 30 days old).")
    if fraud_prob >= reject_th:
        reasons.append(f"LightGBM fraud probability: {fraud_prob*100:.1f}%.")
    if not reasons and decision == "AUTO_APPROVE":
        reasons.append("Transaction aligns with historical behavioral patterns.")
    if not reasons:
        reasons.append(f"Moderate fraud probability ({fraud_prob*100:.1f}%) — manual review recommended.")

    return {
        "fraud_prob": round(fraud_prob, 4),
        "decision":   decision,
        "reasons":    reasons,
    }


# ---------------------------------------------------------------------------
# Z-Score fallback
# ---------------------------------------------------------------------------

def predict_with_zscore(tx: TransactionInput) -> dict:
    """Fallback when LightGBM model is not loaded."""
    avg = tx.average_amount or 0.0
    std = 0.0

    if active_profiles is not None and not active_profiles.empty:
        mask = active_profiles["sender_company_name"] == tx.sender_company_name
        if mask.any():
            row = active_profiles[mask].iloc[0]
            avg = float(row["avg_amount"])
            std = float(row["std_amount"])

    anomaly_score = 50.0
    if std > 0:
        z = abs((tx.amount - avg) / std)
        anomaly_score = min((z / 3.0) * 100.0, 100.0)
    elif avg > 0:
        ratio = tx.amount / avg
        anomaly_score = min((ratio - 1.0) * 50.0, 100.0) if ratio > 1 else 0.0

    final = round((tx.risk_score * 0.6) + (anomaly_score * 0.4), 2)

    if final >= 75:
        decision = "AUTO_REJECT"
    elif final <= 30:
        decision = "AUTO_APPROVE"
    else:
        decision = "MANUAL_REVIEW"

    reasons = []
    if anomaly_score > 75:
        reasons.append(f"Amount (${tx.amount:,.2f}) deviates significantly from historical average.")
    if tx.risk_score > 70:
        reasons.append("High base risk flags from OCR invoice analysis.")
    if not reasons and decision == "AUTO_APPROVE":
        reasons.append("Transaction aligns with standard behavioral patterns.")
    if not reasons:
        reasons.append("Moderate risk level — requires manual review.")

    return {"fraud_prob": None, "decision": decision, "reasons": reasons, "final_score": final}


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/process_transaction", response_model=TransactionDecision)
def process_transaction(tx: TransactionInput):
    """
    Primary fraud detection endpoint.
    Uses LightGBM if model is loaded, otherwise falls back to Z-Score.
    """
    if lgbm_model is not None and model_artifacts is not None:
        # ── LightGBM path ──────────────────────────────────────────────────
        result = predict_with_lgbm(tx)
        final_score = round(result["fraud_prob"] * 100, 2)

        return TransactionDecision(
            transaction_id=tx.Id,
            final_risk_score=final_score,
            fraud_probability=result["fraud_prob"],
            decision=result["decision"],
            reasons=result["reasons"],
            engine="LightGBM",
        )
    else:
        # ── Z-Score fallback ───────────────────────────────────────────────
        result = predict_with_zscore(tx)

        return TransactionDecision(
            transaction_id=tx.Id,
            final_risk_score=result.get("final_score", 50.0),
            fraud_probability=None,
            decision=result["decision"],
            reasons=result["reasons"],
            engine="Z-Score (fallback)",
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    profile_count = len(active_profiles) if active_profiles is not None else 0
    return {
        "status":          "ok",
        "engine":          "LightGBM" if lgbm_model is not None else "Z-Score",
        "profiles_loaded": profile_count,
        "model_loaded":    lgbm_model is not None,
        "auc_score":       model_artifacts.get("auc_score") if model_artifacts else None,
    }