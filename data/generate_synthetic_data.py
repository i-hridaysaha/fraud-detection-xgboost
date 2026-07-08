"""
generate_synthetic_data.py

Generates a synthetic transaction dataset that is schema-compatible with the
Kaggle "Credit Card Fraud Detection" dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud)
— i.e. it includes Time, V1-V28 (PCA-style anonymized features), Amount, and
Class — but AUGMENTS it with customer_id, merchant_id, merchant_category,
device_id and a real timestamp column. The public Kaggle dataset is fully
anonymized and does NOT contain these identity fields, so velocity, merchant
risk, and device-consistency features cannot be built from it directly.

USE WITH REAL DATA:
If you have downloaded creditcard.csv from Kaggle, you can still run the
full pipeline: src/data_loader.py will detect a plain Kaggle-schema file
(no customer_id/merchant_id/device_id columns) and synthesize those identity
fields on top of the real V1-V28/Amount/Class values, using the same logic
below (see `attach_identity_fields`). This keeps the fraud signal genuine
(from the real dataset) while still enabling the full feature set.

Run:
    python data/generate_synthetic_data.py --n_rows 500000 --fraud_rate 0.0017
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def attach_identity_fields(df: pd.DataFrame, n_customers: int, n_merchants: int,
                            n_devices: int, seed: int = 42) -> pd.DataFrame:
    """Adds customer_id, merchant_id, merchant_category, device_id, timestamp
    to a dataframe that already has Time/Amount/Class (Kaggle schema or synthetic).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Customers follow a power-law-ish usage pattern: some customers transact far more than others.
    customer_weights = rng.pareto(a=2.0, size=n_customers) + 0.1
    customer_weights /= customer_weights.sum()
    df["customer_id"] = rng.choice(
        [f"CUST_{i:06d}" for i in range(n_customers)], size=n, p=customer_weights
    )

    merchant_categories = [
        "grocery", "electronics", "travel", "gambling", "crypto_exchange",
        "restaurant", "utilities", "clothing", "pharmacy", "online_marketplace",
    ]
    # gambling/crypto/electronics carry structurally higher fraud risk
    category_fraud_bias = {
        "grocery": 0.2, "electronics": 1.5, "travel": 1.2, "gambling": 3.0,
        "crypto_exchange": 3.5, "restaurant": 0.2, "utilities": 0.1,
        "clothing": 0.6, "pharmacy": 0.3, "online_marketplace": 1.8,
    }
    df["merchant_id"] = rng.integers(0, n_merchants, size=n)
    merchant_to_category = {
        m: merchant_categories[m % len(merchant_categories)] for m in range(n_merchants)
    }
    df["merchant_category"] = df["merchant_id"].map(merchant_to_category)
    df["merchant_fraud_bias"] = df["merchant_category"].map(category_fraud_bias)

    # Devices: most customers use 1-2 devices consistently; a new/unrecognized
    # device is itself a fraud signal we encode via device_id randomness.
    df["device_id"] = [
        f"DEV_{rng.integers(0, max(2, n_devices // n_customers)):03d}_{cust}"
        if rng.random() > 0.03  # 3% chance of a brand-new device (higher risk)
        else f"DEV_NEW_{rng.integers(0, 999999):06d}"
        for cust in df["customer_id"]
    ]

    base_time = datetime(2024, 1, 1)
    df["timestamp"] = [base_time + timedelta(seconds=int(t)) for t in df["Time"]]

    return df


def generate_synthetic_kaggle_style(n_rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    n_pca = 28
    # Legit transactions: PCA features centered near 0, tight spread
    legit_pca = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_pca))
    # Fraud transactions: shifted/heavier-tailed on a subset of components,
    # mimicking known structure in the real dataset (fraud separates on a few PCs)
    fraud_pca = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, n_pca))
    shifted_components = [1, 3, 4, 10, 12, 14, 17]
    for c in shifted_components:
        fraud_pca[:, c] += rng.choice([-1, 1]) * rng.uniform(2.5, 5.0)

    legit_amount = np.round(np.abs(rng.lognormal(mean=3.0, sigma=1.1, size=n_legit)), 2)
    fraud_amount = np.round(np.abs(rng.lognormal(mean=4.2, sigma=1.4, size=n_fraud)), 2)

    legit_time = np.sort(rng.uniform(0, 60 * 60 * 24 * 30, size=n_legit))  # 30 days
    fraud_time = np.sort(rng.uniform(0, 60 * 60 * 24 * 30, size=n_fraud))

    df_legit = pd.DataFrame(legit_pca, columns=[f"V{i+1}" for i in range(n_pca)])
    df_legit["Time"] = legit_time
    df_legit["Amount"] = legit_amount
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(fraud_pca, columns=[f"V{i+1}" for i in range(n_pca)])
    df_fraud["Time"] = fraud_time
    df_fraud["Amount"] = fraud_amount
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=300_000)
    parser.add_argument("--fraud_rate", type=float, default=0.0017)  # ~ Kaggle's real-world rate
    parser.add_argument("--n_customers", type=int, default=20_000)
    parser.add_argument("--n_merchants", type=int, default=3_000)
    parser.add_argument("--n_devices", type=int, default=25_000)
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_rows:,} synthetic transactions "
          f"(fraud rate ~{args.fraud_rate:.4%}) ...")
    df = generate_synthetic_kaggle_style(args.n_rows, args.fraud_rate, args.seed)
    df = attach_identity_fields(df, args.n_customers, args.n_merchants, args.n_devices, args.seed)

    # Fraud is more likely on high-risk merchant categories and new devices —
    # bias a fraction of Class labels to correlate with those fields so the
    # engineered features actually carry signal (rather than being pure noise).
    rng = np.random.default_rng(args.seed)
    is_new_device = df["device_id"].str.startswith("DEV_NEW_")
    boost_mask = (rng.random(len(df)) < (0.15 * df["merchant_fraud_bias"] / df["merchant_fraud_bias"].max())) | \
                 (is_new_device & (rng.random(len(df)) < 0.05))
    flip_to_fraud = boost_mask & (df["Class"] == 0) & (rng.random(len(df)) < 0.4)
    df.loc[flip_to_fraud, "Class"] = 1

    df = df.drop(columns=["merchant_fraud_bias"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Fraud count: {df['Class'].sum():,} ({df['Class'].mean():.4%})")


if __name__ == "__main__":
    main()
