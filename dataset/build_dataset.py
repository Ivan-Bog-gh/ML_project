import pandas as pd


def build_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    suspend_flag: pd.Series,
    max_lookback: int,
) -> pd.DataFrame:
    """
    Assemble final modeling dataset.

    Steps:
    1. Join features and labels by timestamp
    2. Drop warmup period (rolling lookback)
    3. Exclude samples contaminated by suspended data
    """

    # --- 1. join ---
    df = features.join(labels.rename("label"), how="inner")

    # --- 2. drop warmup ---
    df = df.iloc[max_lookback:]

    # --- 3. suspend mask ---
    suspend_mask = suspend_flag.rolling(max_lookback).max().reindex(df.index).fillna(1)
    print(f"Number of suspended samples: {sum(suspend_mask == 1)}")
    df = df[suspend_mask == 0]

    
    # --- 4. drop label=2 ---
    label_2_count = (df["label"] == 2).sum()
    if label_2_count > 0:
        print(f"Удаление {label_2_count:,} строк с label=2...")
        df = df[df["label"] != 2]

    # --- 5. drop NaN ---
    nan_count = df.isna().any(axis=1).sum()
    if nan_count > 0:
        print(f"Удаление {nan_count:,} строк с NaN...")
        df = df.dropna()

    # --- 6. sanity ---
    assert "label" in df.columns
    assert df.isna().sum().sum() == 0, "NaNs detected in final dataset"

    return df
