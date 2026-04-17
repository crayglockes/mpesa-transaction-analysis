"""
feature_engineering.py
----------------------
Creates user-level and transaction-level features for
downstream ML tasks (segmentation + anomaly detection).
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ── Transaction-Level Features ────────────────────────────────────────────────

def add_temporal_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    df = txn_df.copy()
    df['hour']           = df['timestamp'].dt.hour
    df['day_of_week']    = df['timestamp'].dt.dayofweek
    df['is_weekend']     = df['day_of_week'] >= 5
    df['is_night']       = (df['hour'] >= 22) | (df['hour'] < 5)
    df['is_rush_hour']   = df['hour'].isin([7,8,9,17,18,19])
    df['month']          = df['timestamp'].dt.month
    df['quarter']        = df['timestamp'].dt.quarter
    df['is_month_end']   = df['timestamp'].dt.is_month_end
    df['is_month_start'] = df['timestamp'].dt.is_month_start
    df['is_salary_week'] = df['timestamp'].dt.day <= 7
    return df


def add_amount_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    df = txn_df.copy()
    df['log_amount']          = np.log1p(df['amount_kes'])
    df['amount_to_fee_ratio'] = df['amount_kes'] / (df['fee_kes'] + 1)
    df['is_round_amount']     = (df['amount_kes'] % 1000 == 0) & (df['amount_kes'] >= 5000)

    THRESHOLDS = [1000, 5000, 10000, 35000, 70000, 150000]
    df['is_just_below_threshold'] = False
    for thresh in THRESHOLDS:
        df['is_just_below_threshold'] |= (
            (df['amount_kes'] >= thresh * 0.95) & (df['amount_kes'] < thresh)
        )
    return df


# ── User-Level Aggregated Features ───────────────────────────────────────────

def build_user_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction history into per-user feature vectors.
    The total_sent column is computed with a pre-filter to avoid capturing
    the outer df reference inside a lambda — a subtle but real closure bug.
    """
    df = add_temporal_features(txn_df)
    df = add_amount_features(df)

    # Pre-filter send_money before groupby — avoids lambda capturing outer scope
    send_money_mask = df['txn_type'] == 'send_money'

    base = df.groupby('user_id').agg(
        txn_count        = ('txn_id',    'count'),
        total_value      = ('amount_kes','sum'),
        avg_amount       = ('amount_kes','mean'),
        median_amount    = ('amount_kes','median'),
        std_amount       = ('amount_kes','std'),
        max_amount       = ('amount_kes','max'),
        min_amount       = ('amount_kes','min'),
        total_fees_paid  = ('fee_kes',   'sum'),
        unique_txn_types = ('txn_type',  'nunique'),
    )

    # Compute total_sent separately and join (avoids closure over outer `df`)
    total_sent = (
        df[send_money_mask]
        .groupby('user_id')['amount_kes']
        .sum()
        .rename('total_sent')
    )
    base = base.join(total_sent, how='left').fillna({'total_sent': 0})

    # Behavioural type ratios
    type_counts = df.groupby(['user_id','txn_type']).size().unstack(fill_value=0)
    for col in type_counts.columns:
        type_counts[col] = type_counts[col] / type_counts.sum(axis=1)
        type_counts.rename(columns={col: f'pct_{col}'}, inplace=True)

    temporal = df.groupby('user_id').agg(
        pct_night_txns  = ('is_night',      'mean'),
        pct_weekend_txns= ('is_weekend',    'mean'),
        pct_rush_hour   = ('is_rush_hour',  'mean'),
        pct_salary_week = ('is_salary_week','mean'),
        pct_month_end   = ('is_month_end',  'mean'),
        unique_hours    = ('hour',          'nunique'),
        preferred_hour  = ('hour',          lambda x: x.mode()[0]),
    )

    amount_dist = df.groupby('user_id').agg(
        pct_round_amounts         = ('is_round_amount',        'mean'),
        pct_just_below_threshold  = ('is_just_below_threshold','mean'),
        log_amount_mean           = ('log_amount',             'mean'),
        log_amount_std            = ('log_amount',             'std'),
    )

    activity = df.groupby('user_id')['timestamp'].agg(['min','max'])
    activity['active_days'] = (activity['max'] - activity['min']).dt.days + 1
    activity = activity[['active_days']]

    features = base.join(type_counts,   how='left')
    features = features.join(temporal,  how='left')
    features = features.join(amount_dist, how='left')
    features = features.join(activity,  how='left')

    features['txn_per_day']    = features['txn_count'] / features['active_days'].clip(1)
    features['fees_pct_value'] = features['total_fees_paid'] / (features['total_value'] + 1)
    features['value_per_txn']  = features['total_value'] / features['txn_count'].clip(1)
    features['amount_cv']      = features['std_amount'] / (features['avg_amount'] + 1)

    return features.fillna(0)


def build_fraud_labels(txn_df: pd.DataFrame) -> pd.Series:
    """User-level fraud label: 1 if any transaction flagged."""
    return txn_df.groupby('user_id')['is_fraud_flag'].max().rename('is_fraud')


if __name__ == "__main__":
    import os
    txn_df   = pd.read_parquet('data/raw/transactions.parquet')
    features = build_user_features(txn_df)
    labels   = build_fraud_labels(txn_df)

    print(f"Feature matrix: {features.shape}")
    print(f"Fraud rate:     {labels.mean()*100:.1f}%")

    os.makedirs('data/processed', exist_ok=True)
    features.to_parquet('data/processed/user_features.parquet')
    labels.to_frame().to_parquet('data/processed/fraud_labels.parquet')
    print("✅ Saved to data/processed/")
