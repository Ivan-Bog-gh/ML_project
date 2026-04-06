# dataset/splits.py

from typing import Iterator, Tuple
import pandas as pd


def time_split(
    df: pd.DataFrame,
    time_periods: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple time-based train/validation/etc. split.
    """
    start_time = df.index.min()
    periods = []
    for t in time_periods:
        assert t in df.index, f"Time period {t} not found in DataFrame index"
        periods.append(df[(df.index < t) & (df.index >= start_time)])
        start_time = t
    periods.append(df[df.index >= time_periods[-1]])

    assert all(len(period) > 0 for period in periods), "Empty split detected"

    return periods


def walk_forward_split(
    df: pd.DataFrame,
    splits: list[tuple[str, str]],
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward validation generator.

    splits: [(train_end, val_end), ...]
    """

    for train_end, val_end in splits:
        train = df[df.index < train_end]
        val = df[(df.index >= train_end) & (df.index < val_end)]

        if len(train) == 0 or len(val) == 0:
            continue

        yield train, val
