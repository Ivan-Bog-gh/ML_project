from typing import Iterator, Tuple
import pandas as pd


def time_split(
    df: pd.DataFrame,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple time-based train/validation split.
    """

    train = df[df.index < split_date]
    val = df[df.index >= split_date]

    assert len(train) > 0 and len(val) > 0, "Empty split detected"

    return train, val


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
