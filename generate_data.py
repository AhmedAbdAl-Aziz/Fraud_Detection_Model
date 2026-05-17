"""
generate_data.py — Realistic Synthetic Fraud Dataset Generator
==============================================================

ينتج ملف 'synthetic_transactions.csv' بـ 50,000 صف
مع patterns حقيقية للـ fraud بحيث يستفيد منها LightGBM.

Fraud patterns مضمّنة:
  1. مبلغ أكبر من 3x متوسط الشركة (anomaly amount)
  2. risk_score عالي من الـ OCR
  3. شركة جديدة بمبلغ كبير (new entity violation)
  4. معاملة في ساعات غريبة (outside working hours)
  5. Z-score عالي (deviation من الـ behavioral profile)
  6. مجموعة عوامل متوسطة (moderate combined risk)

الاستخدام:
  python generate_data.py
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
N_TOTAL     = 50_000
FRAUD_RATIO = 0.25          # 25% fraud, 75% legitimate
OUTPUT_CSV  = "synthetic_transactions.csv"

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Company Profiles (50 companies across 6 industries)
# ─────────────────────────────────────────────────────────────────────────────
COMPANIES = [
    # name,               industry,        avg_tx,   std_tx,  age_days
    ("AlphaTech Ltd",      "Technology",    15_000,   4_000,   730),
    ("BetaCorp",           "Technology",    22_000,   6_500,   540),
    ("GammaSoft",          "Technology",     8_000,   2_000,   365),
    ("DeltaAI",            "Technology",    50_000,  15_000,   180),
    ("EpsilonData",        "Technology",    12_000,   3_000,   900),
    ("ZetaCloud",          "Technology",    35_000,   9_000,   270),

    ("NileTrading",        "Trading",       45_000,  12_000,   600),
    ("CairoExport",        "Trading",       80_000,  20_000,   1200),
    ("MedPort Ltd",        "Trading",       30_000,   8_000,   450),
    ("GulfMerchants",      "Trading",      120_000,  35_000,   800),
    ("SuezCommerce",       "Trading",       60_000,  18_000,   360),
    ("AlexImport",         "Trading",       25_000,   7_000,   720),

    ("HealthFirst",        "Healthcare",    18_000,   5_000,   550),
    ("MedSupply Co",       "Healthcare",    40_000,  11_000,   400),
    ("PharmaLink",         "Healthcare",    75_000,  22_000,   900),
    ("ClinicNet",          "Healthcare",    10_000,   3_000,   200),
    ("BioTrade",           "Healthcare",    55_000,  16_000,   650),

    ("BuildPro",           "Construction",  90_000,  25_000,   1000),
    ("SteelMasters",       "Construction", 200_000,  60_000,   800),
    ("CementGroup",        "Construction", 150_000,  45_000,   600),
    ("ArchBuild",          "Construction",  35_000,  10_000,   300),
    ("InfraCore",          "Construction", 500_000, 150_000,   1500),

    ("EduTech",            "Education",      5_000,   1_500,   400),
    ("LearnHub",           "Education",      8_000,   2_000,   300),
    ("AcademyPro",         "Education",     12_000,   3_500,   600),
    ("SkillsNet",          "Education",      3_000,     800,   150),

    ("FoodDist",           "Food",          25_000,   7_000,   700),
    ("FreshMarket",        "Food",          15_000,   4_000,   500),
    ("AgriTrade",          "Food",          40_000,  12_000,   600),
    ("HarvestCo",          "Food",          60_000,  18_000,   800),
    ("GrainExport",        "Food",          80_000,  24_000,   900),

    # NEW companies (high risk — low age_days)
    ("NewTech2024",        "Technology",    10_000,   3_000,    15),
    ("StartupX",           "Trading",       50_000,  15_000,     5),
    ("QuickPay Ltd",       "Trading",       30_000,   9_000,    30),
    ("FastTrade",          "Trading",       20_000,   6_000,    20),
    ("MicroVentures",      "Technology",     5_000,   1_500,    10),

    # High-value companies
    ("MegaCorp",           "Trading",      500_000, 150_000,   2000),
    ("GlobalTrade Inc",    "Trading",      300_000,  90_000,   1800),
    ("OilLink",            "Trading",    1_000_000, 300_000,   3000),

    # Moderate companies
    ("RegioBank",          "Finance",       70_000,  20_000,   1000),
    ("FinServe",           "Finance",       90_000,  27_000,   800),
    ("WealthMgmt",         "Finance",      200_000,  60_000,   1200),

    ("LogiTrans",          "Logistics",     30_000,   9_000,   600),
    ("ShipFast",           "Logistics",     45_000,  13_500,   500),
    ("CargoNet",           "Logistics",     55_000,  16_500,   700),

    ("MediaHub",           "Media",          8_000,   2_400,   400),
    ("AdVision",           "Media",         15_000,   4_500,   500),
    ("ContentCo",          "Media",          5_000,   1_500,   300),

    ("SolarEnergy",        "Energy",        80_000,  24_000,   600),
    ("PowerGrid",          "Energy",       200_000,  60_000,   900),
]

company_df = pd.DataFrame(
    COMPANIES,
    columns=["company_name", "industry", "avg_tx", "std_tx", "age_days"],
)
company_df.index.name = "company_id"
company_df = company_df.reset_index()
company_df["company_id"] += 1  # 1-based

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def pick_companies(n):
    """Pick n sender/receiver pairs (different companies)."""
    sender_ids   = rng.integers(0, len(company_df), size=n)
    receiver_ids = rng.integers(0, len(company_df), size=n)
    # Ensure sender != receiver
    same = sender_ids == receiver_ids
    receiver_ids[same] = (receiver_ids[same] + 1) % len(company_df)
    return sender_ids, receiver_ids


def compute_risk_score(amount, avg, std, age_days, hour, is_fraud_hint):
    """Compute a realistic OCR risk_score correlated with fraud indicators."""
    score = 0.0

    # Amount anomaly
    if std > 0:
        z = (amount - avg) / std
        score += min(z * 10, 40) if z > 0 else 0

    # New entity
    if age_days < 30:
        score += 25
    elif age_days < 90:
        score += 10

    # Odd hours (11pm-5am)
    if hour >= 23 or hour <= 5:
        score += 15

    # Large absolute amount
    if amount > 500_000:
        score += 10
    elif amount > 200_000:
        score += 5

    # Add noise
    noise = rng.normal(0, 5)
    score = np.clip(score + noise, 0, 100)

    # Fraud hint: push score up
    if is_fraud_hint:
        score = np.clip(score + rng.uniform(10, 30), 0, 100)
    else:
        score = np.clip(score - rng.uniform(0, 10), 0, 100)

    return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate Records
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Synthetic Fraud Dataset Generator")
print("=" * 60)
print(f"\nGenerating {N_TOTAL:,} transactions ({int(FRAUD_RATIO*100)}% fraud) …")

n_fraud = int(N_TOTAL * FRAUD_RATIO)
n_legit = N_TOTAL - n_fraud

records = []

# ── Legitimate transactions ─────────────────────────────────────────────────
s_ids, r_ids = pick_companies(n_legit)

for i in range(n_legit):
    s = company_df.iloc[s_ids[i]]
    r = company_df.iloc[r_ids[i]]

    amount = abs(rng.normal(s["avg_tx"], s["std_tx"] * 0.8))
    amount = round(max(amount, 10.0), 2)

    hour = int(rng.integers(8, 20))  # Business hours

    records.append({
        "sender_company_name":   s["company_name"],
        "receiver_company_name": r["company_name"],
        "industry":              s["industry"],
        "amount":                amount,
        "hour_of_day":           hour,
        "is_weekend":            int(rng.random() < 0.1),
        "sender_age_days":       s["age_days"],
        "sender_avg_amount":     s["avg_tx"],
        "sender_std_amount":     s["std_tx"],
        "sender_tx_count":       int(rng.integers(5, 500)),
        "risk_score":            compute_risk_score(amount, s["avg_tx"], s["std_tx"], s["age_days"], hour, False),
        "label":                 0,  # Legitimate
        "fraud_type":            "none",
    })

# ── Fraudulent transactions (6 patterns) ────────────────────────────────────
n_per_pattern = n_fraud // 6
remainder      = n_fraud % 6

patterns = [
    ("amount_anomaly",     n_per_pattern),
    ("high_risk_score",    n_per_pattern),
    ("new_entity",         n_per_pattern),
    ("odd_hours",          n_per_pattern),
    ("structuring",        n_per_pattern),
    ("combined_moderate",  n_per_pattern + remainder),
]

for fraud_type, count in patterns:
    s_ids, r_ids = pick_companies(count)

    for i in range(count):
        s = company_df.iloc[s_ids[i]]
        r = company_df.iloc[r_ids[i]]

        # Base amount = normal
        base_amount = abs(rng.normal(s["avg_tx"], s["std_tx"] * 0.5))

        if fraud_type == "amount_anomaly":
            # Amount is 3x-10x the average → clear anomaly
            multiplier = rng.uniform(3.0, 10.0)
            amount     = base_amount * multiplier
            hour       = int(rng.integers(8, 20))

        elif fraud_type == "high_risk_score":
            # Amount is large, risk_score will be high
            amount = base_amount * rng.uniform(2.5, 6.0)
            hour   = int(rng.integers(8, 20))

        elif fraud_type == "new_entity":
            # Force new company doing large transaction
            new_companies = company_df[company_df["age_days"] < 30]
            s = new_companies.iloc[rng.integers(0, len(new_companies))]
            amount = abs(rng.normal(s["avg_tx"], s["std_tx"])) * rng.uniform(2.0, 5.0)
            hour   = int(rng.integers(8, 20))

        elif fraud_type == "odd_hours":
            # Transaction at 11pm-5am
            hour   = int(rng.choice([23, 0, 1, 2, 3, 4, 5]))
            amount = base_amount * rng.uniform(1.5, 3.0)

        elif fraud_type == "structuring":
            # Just below round-number threshold (structuring / smurfing)
            thresholds = [10_000, 50_000, 100_000, 500_000]
            threshold  = rng.choice(thresholds)
            amount     = threshold - rng.uniform(1, 500)
            hour       = int(rng.integers(8, 20))

        elif fraud_type == "combined_moderate":
            # Moderate on multiple dimensions
            amount = base_amount * rng.uniform(1.8, 3.5)
            hour   = int(rng.choice([6, 7, 21, 22, 23]))

        amount = round(max(float(amount), 10.0), 2)

        records.append({
            "sender_company_name":   s["company_name"],
            "receiver_company_name": r["company_name"],
            "industry":              s["industry"],
            "amount":                amount,
            "hour_of_day":           hour,
            "is_weekend":            int(rng.random() < 0.25),
            "sender_age_days":       s["age_days"],
            "sender_avg_amount":     s["avg_tx"],
            "sender_std_amount":     s["std_tx"],
            "sender_tx_count":       int(rng.integers(1, 50)),
            "risk_score":            compute_risk_score(amount, s["avg_tx"], s["std_tx"], s["age_days"], hour, True),
            "label":                 1,  # Fraud
            "fraud_type":            fraud_type,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
df.index.name = "Id"
df = df.reset_index()
df["Id"] = "TX-" + df["Id"].astype(str).str.zfill(5)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n  Total rows : {len(df):,}")
print(f"  Fraud rows : {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")
print(f"  Legit rows : {(df['label']==0).sum():,}")
print(f"\n  Fraud types distribution:")
print(df[df['label']==1]['fraud_type'].value_counts().to_string())
print(f"\n  Unique senders   : {df['sender_company_name'].nunique()}")
print(f"  Unique receivers : {df['receiver_company_name'].nunique()}")
print(f"\n✅ Dataset saved → '{OUTPUT_CSV}'")
