# features/feature_cleaning.py

import numpy as np
import pandas as pd


# ─── HELPERS ───────────────────────────────────────────────────────────────

def drop_near_constant(df, min_std=1e-6):
    stds = df.std()
    keep = stds[stds > min_std].index
    throw = stds[stds <= min_std].index
    if len(throw) > 0:
        print(f"Dropping {df.shape[1] - len(keep)} near-constant features:  {list(throw)}")
    return df[keep]


def drop_correlated(df, threshold=0.98):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    if drop_cols:
        print(f"Dropping {len(drop_cols)} highly correlated features:  {' / '.join(list(drop_cols))}")
    return df.drop(columns=drop_cols)


def check_feature_label_corr(df_features, labels):
    # df_features = df_features.reset_index(drop=True)
    corr = df_features.corrwith(labels).abs().sort_values(ascending=False)
    suspicious = corr[corr > 0.3]

    if not suspicious.empty:
        print("WARNING: high feature-label correlation")
        print(suspicious.head(5))

def clean_features(df_features, labels):
    df = drop_near_constant(df_features)
    df = drop_correlated(df)
    check_feature_label_corr(df, labels)
    return df
