"""
data_prep.py — Build behavioral profiles for the Fraud Detection Engine.

Two modes:
  1. FROM CSV  (default / offline)
     python data_prep.py
     Reads 'transaction master.csv' and builds sender_profiles.pkl.

  2. FROM BACKEND API  (production / online)
     python data_prep.py --from-api
     Calls the Laravel backend to fetch real transaction data and builds profiles.
     Requires environment variables:
       BACKEND_URL   — e.g. https://b2bshield.dpdns.org
       BACKEND_TOKEN — Admin JWT token (generate once, store as Railway secret)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Core profile builder
# ---------------------------------------------------------------------------

def build_behavior_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with at least:
        sender_company_name, amount
    Returns a profile DataFrame grouped by sender.
    """
    df.columns = df.columns.str.strip().str.lower()

    # Normalise amount column
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(" USD", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

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
    return profiles


# ---------------------------------------------------------------------------
# Mode 1: Build from CSV
# ---------------------------------------------------------------------------

def build_from_csv(csv_path: str = "transaction master.csv") -> pd.DataFrame:
    print(f"📂 Loading dataset from '{csv_path}' …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Rows loaded: {len(df):,}")
    profiles = build_behavior_profiles(df)
    print(f"   Companies profiled: {len(profiles)}")
    return profiles


# ---------------------------------------------------------------------------
# Mode 2: Build from Backend API
# ---------------------------------------------------------------------------

def build_from_api(backend_url: str, token: str) -> pd.DataFrame:
    """
    Fetches ALL transactions from the Laravel admin endpoint
    and builds behavior profiles from real data.
    """
    print(f"🌐 Fetching transactions from {backend_url} …")

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    all_rows = []
    page = 1
    limit = 100

    while True:
        resp = requests.get(
            f"{backend_url}/api/admin/transactions",
            headers=headers,
            params={"limit": limit, "page": page},
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"❌ API error {resp.status_code}: {resp.text[:300]}")
            sys.exit(1)

        data = resp.json()
        # Laravel paginator shape: {data: [...], last_page: N}
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            all_rows.append(
                {
                    "sender_company_name": item.get("sender_name", "Unknown"),
                    "amount": item.get("amount", 0),
                }
            )

        current_page = data.get("current_page", page)
        last_page = data.get("last_page", 1)
        print(f"   Page {current_page}/{last_page} — {len(all_rows):,} rows so far")

        if current_page >= last_page:
            break
        page += 1

    if not all_rows:
        print("⚠️  No transactions returned from the API.")
        return pd.DataFrame(
            columns=["sender_company_name", "avg_amount", "std_amount", "transaction_count"]
        )

    df = pd.DataFrame(all_rows)
    profiles = build_behavior_profiles(df)
    print(f"   Companies profiled: {len(profiles)}")
    return profiles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build sender behavior profiles.")
    parser.add_argument(
        "--from-api",
        action="store_true",
        help="Fetch transactions from the Laravel backend API instead of local CSV.",
    )
    parser.add_argument(
        "--csv",
        default="transaction master.csv",
        help="Path to the CSV file (used when --from-api is not set).",
    )
    parser.add_argument(
        "--output",
        default="sender_profiles.pkl",
        help="Output pickle path.",
    )
    args = parser.parse_args()

    if args.from_api:
        backend_url = os.environ.get("BACKEND_URL", "https://b2bshield.dpdns.org")
        token = os.environ.get("BACKEND_TOKEN", "")
        if not token:
            print("❌ BACKEND_TOKEN environment variable is required for --from-api mode.")
            sys.exit(1)
        profiles = build_from_api(backend_url, token)
    else:
        profiles = build_from_csv(args.csv)

    profiles.to_pickle(args.output)
    print(f"\n✅ Profiles saved to '{args.output}'")
    print(profiles.to_string(index=False))