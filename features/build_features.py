# features/build_features.py

import numpy as np
import pandas as pd
from pathlib import Path

import sys
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from features.base_features import parallel_compute
from features.feature_cleaning import clean_features


INTERIM_DIR     = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"


# ─── HELPERS ───────────────────────────────────────────────────────────────

def compute_special_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет базовые микроструктурные признаки.
    Ожидает колонки: 'open', 'close', 'volume', 'taker_buy_base', 'trades_count'.
    """
    df = df.copy()
    
    # 1. Дельта объёма (агрессивные покупки минус продажи)
    df['volume_delta'] = 2 * df['taker_buy_base'] - df['volume']
    
    # 2. Дисбаланс объёма (с защитой от деления на ноль)
    df['volume_imbalance'] = df['volume_delta'] / df['volume'].replace(0, np.nan)
    
    # 3. Кумулятивная дельта (CVD)
    df['cvd'] = df['volume_delta'].fillna(0).cumsum()
    
    # 4. Средний размер сделки
    df['avg_trade_size'] = df['volume'] / df['trades_count'].replace(0, np.nan)
    
    # 5. Прокси изменения открытого интереса (OI) — предполагая, что OI растёт при агрессивных покупках/продажах и падает при пассивных сделках
    df['oi_proxy_delta'] = df['volume_delta'] * np.sign(df['close'] - df['open'])
    
    # 6. Кумулятивный прокси OI
    df['oi_proxy'] = df['oi_proxy_delta'].fillna(0).cumsum()
    
    return df


# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      help="путь к .parquet с OHLCV")
    parser.add_argument("--output",     help="куда сохранить features .parquet")
    parser.add_argument("--labels",     help="Где лежат labels .parquet")
    parser.add_argument("--n-jobs",     type=int, default=-1,   help="число процессов (-1 = все кроме одного)")
    parser.add_argument("--no-args",    action="store_true",    help="Использовать значения по умолчанию без argparse")
    
    args = parser.parse_args()

    if args.no_args:
        symbol = "BTCUSDT"
        tf = "dib_temp"
        path_in     = INTERIM_DIR / f"{symbol}_{tf}.parquet"
        path_out    = PROCESSED_DIR / f"{symbol}_{tf}_features.parquet"
        labels      = PROCESSED_DIR / f"{symbol}_{tf}_labels.parquet"
        n_jobs      = -1
    else:
        path_in     = Path(args.input)
        path_out    = Path(args.output)
        labels      = Path(args.labels)
        n_jobs      = args.n_jobs

    path_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {path_in} ...")
    df = pd.read_parquet(path_in)
    df = compute_special_features(df)  # добавляю фичи до параллельного расчёта

    print(f"Computing features using {n_jobs} jobs ...")
    features = parallel_compute(df, n_jobs=n_jobs)


    # объединяем с исходными данными (опционально)
    # result = pd.concat([df[["open","high","low","close","volume","suspended_flg"]], features], axis=1)
    result = features  # или с исходными колонками — как тебе удобнее

    labels = pd.read_parquet(labels)
    result = clean_features(result, labels=labels)

    print(f"Saving → {path_out}")
    result.to_parquet(path_out, compression="zstd", index=True)
    print("Done.")