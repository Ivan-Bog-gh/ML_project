# labeling/targets.py

"""
Label definition (event-based, directional):

label = +1:
    Price reaches +TP from entry level (TVX) 
    before reaching -SL within the lookahead window.

label = -1:
    Price reaches -TP from entry level (TVX)
    before reaching +SL within the lookahead window.

label = 0 (no-trade):
    Neither TP nor SL is reached within the lookahead window.

label = +2 (samples dropped):
    Price reaches +TP from entry level (TVX) 
    at the same time as -SL within the lookahead window.

Important notes:
- This is NOT a point forecast.
- Labels represent trade feasibility, not price direction certainty.
- Class imbalance is expected and reflects real market conditions.
"""

import logging
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path


# ─── CONFIG ────────────────────────────────────────────────────────────────

logger = logging.getLogger("parquet_labeling")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR= PROJECT_ROOT / "data" / "processed"


# ─── HELPERS ───────────────────────────────────────────────────────────────

def _label_chunk_optimized(args):
    """
    Оптимизированная обработка: итерация по времени (tl),
    а не по каждой строке полностью
    """
    i_start, i_end, close, high, low, tl, threshold = args
    
    n_rows = i_end - i_start
    labels = np.full(n_rows, np.nan, dtype=np.float32)  # изначально все NaN
    
    # Маска: True = строка ещё не получила финальную метку
    active_mask = np.ones(n_rows, dtype=bool)
    
    # Предвычисляем границы для всех активных строк
    entries = close[i_start:i_end]
    upper_bounds = entries * (1 + threshold)
    lower_bounds = entries * (1 - threshold)
    
    # Итерируемся по шагам времени вперёд
    for t in range(1, tl + 1):
        # Индексы в будущем для каждой строки чанка
        future_indices = np.arange(i_start, i_end) + t
        
        # Проверяем, не вышли ли за пределы массива
        valid_future = future_indices < len(close)
        check_mask = active_mask & valid_future
        
        if not np.any(check_mask):
            break  # все строки либо обработаны, либо вышли за пределы
        
        # Получаем будущие цены только для активных строк
        check_indices = np.where(check_mask)[0]
        future_high = high[future_indices[check_indices]]
        future_low = low[future_indices[check_indices]]
        
        # Проверяем касания границ
        hit_upper = future_high >= upper_bounds[check_indices]
        hit_lower = future_low <= lower_bounds[check_indices]
        
        # Обрабатываем одновременные касания (pos_u == pos_l)
        both_hit = hit_upper & hit_lower
        if np.any(both_hit):
            both_indices = check_indices[both_hit]
            labels[both_indices] = 2
            active_mask[both_indices] = False
        
        # Обрабатываем только верхнее касание
        only_upper = hit_upper & ~hit_lower & active_mask[check_indices]
        if np.any(only_upper):
            upper_indices = check_indices[only_upper]
            labels[upper_indices] = 1
            active_mask[upper_indices] = False
        
        # Обрабатываем только нижнее касание
        only_lower = hit_lower & ~hit_upper & active_mask[check_indices]
        if np.any(only_lower):
            lower_indices = check_indices[only_lower]
            labels[lower_indices] = -1
            active_mask[lower_indices] = False
    
    # Оставшиеся активные строки = не было касаний
    labels[active_mask & (np.arange(i_start, i_end) + tl < len(close))] = 0
    
    return labels


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def compute_event_labels_optimized(
    symbol: str, 
    timeframe: str,
    tl: int = 12,
    threshold: float = 0.003,
    price_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    n_jobs: int = -1
) -> pd.Series:
    """
    Оптимизированная версия с итерацией по времени
    """
    symbol          = symbol.upper()
    timeframe       = timeframe.lower()
    interim_path    = INTERIM_DIR / f"{symbol}_{timeframe}.parquet"
    processed_path  = PROCESSED_DIR / f"{symbol}_{timeframe}_labels.parquet"
    
    if not interim_path.is_file():
        logger.error(f"Файл не найден: {interim_path}")
        return
        
    df = pd.read_parquet(interim_path)
        
    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    n = len(df)
    
    # Размер чанка - баланс между параллелизмом и overhead
    n_cores = n_jobs if n_jobs > 0 else 8
    chunk_size = max(1000, n // n_cores + 1)
    
    chunks = [
        (i, min(i + chunk_size, n), close, high, low, tl, threshold)
        for i in range(0, n, chunk_size)
    ]
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_label_chunk_optimized)(chunk) for chunk in chunks
    )
    
    labels = np.concatenate(results)
    labels[-tl:] = np.nan # Сразу мьючу последние строки
    
    labels_df = pd.DataFrame({'label': labels})
    
    labels_df.to_parquet(
        processed_path,
        index=False,
        compression="zstd",          # или "gzip", "snappy"
        engine="pyarrow",
    )
    logger.info(f"Сохранено {len(labels):,} строк → {processed_path.name}")
    return pd.Series(labels, index=df.index, name='label')
    

# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Инкрементальная загрузка OHLCV с Binance")
    parser.add_argument("--symbol",     default="BTCUSDT",      help="Торговая пара")
    parser.add_argument("--timeframe",  default="5m",           help="Таймфрейм")
    parser.add_argument("--tl",         default=12,             help="Горизонт событий")
    parser.add_argument("--threshold",  default=0.01,          help="Границы определения событий")
    parser.add_argument("--no-args",    action="store_true",    help="Использовать значения по умолчанию без argparse")

    args = parser.parse_args()

    # Если явно попросили игнорировать аргументы → используем старый удобный пример
    if args.no_args:
        symbol      = "BTCUSDT"
        timeframe   = "5m"
        tl          = 12
        threshold   = 0.01
    else:
        symbol      = args.symbol
        timeframe   = args.timeframe
        tl          = args.tl
        threshold   = args.threshold

    compute_event_labels_optimized(symbol=symbol, timeframe=timeframe, tl=tl, threshold=threshold)