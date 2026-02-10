# preprocessing/clean_ohlcv.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальная чистка + отсев явного мусора"""
    original_len = len(df)

    # 1. Типы
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 2. Удаляем строки с NaN в ключевых полях
    df = df.dropna(subset=['open_time'] + numeric_cols)

    # 3. Физическая невозможность
    invalid = (
        (df['high'] < df['low']) |
        (df['high'] < df[['open', 'close']].max(axis=1)) |
        (df['low']  > df[['open', 'close']].min(axis=1)) |
        (df['volume'] < 0) |
        (df[numeric_cols].le(0).all(axis=1))   # все OHLCV ≤ 0 — мусор
    )
    if invalid.any():
        logger.warning(f"Удалено {invalid.sum()} физически невозможных свечей")
        df = df[~invalid]

    # 4. Дедупликация (на всякий случай)
    df = df.drop_duplicates(subset=['open_time'], keep='last')

    logger.info(f"После чистки: {len(df)} строк (было {original_len})")
    return df.sort_values('open_time').reset_index(drop=True)