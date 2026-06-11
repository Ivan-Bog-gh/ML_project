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
import yaml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path


# ─── CONFIG ────────────────────────────────────────────────────────────────

logger = logging.getLogger("parquet_labeling")

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
INTERIM_DIR     = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR      = PROJECT_ROOT / "config"


# ─── HELPERS ───────────────────────────────────────────────────────────────

def _label_chunk_optimized(args):
    """
    Оптимизированная обработка Triple Barrier для DIB.
    
    Горизонт задаётся не числом баров (tl), а временем (horizon_seconds),
    поэтому для каждой строки i_start..i_end вычисляем индивидуальный
    максимальный future-индекс через timestamps.
    
    Labels:
         1  — hit upper barrier (TP)
        -1  — hit lower barrier (SL)
         2  — hit both simultaneously
         0  — горизонт истёк без касания
        NaN — горизонт вышел за пределы массива (недостаточно данных)
    """
    (
        i_start,
        i_end,
        close,       # np.ndarray float32, все бары
        high,        # np.ndarray float32, все бары
        low,         # np.ndarray float32, все бары
        timestamps,  # np.ndarray float64 (unix seconds), все бары
        tp_thresholds,  # np.ndarray float32, индивидуальный TP % для каждого бара
        sl_thresholds,  # np.ndarray float32, индивидуальный SL % для каждого бара
        horizon_seconds,  # int/float: временной горизонт в секундах
    ) = args

    n_rows = i_end - i_start
    labels = np.full(n_rows, np.nan, dtype=np.float32)

    entries        = close[i_start:i_end]
    upper_bounds   = entries * (1 + tp_thresholds[i_start:i_end])
    lower_bounds   = entries * (1 - sl_thresholds[i_start:i_end])
    entry_times    = timestamps[i_start:i_end]

    # Для каждой строки — максимальный индекс будущего бара в горизонте
    # np.searchsorted быстрее, чем цикл
    horizon_end_times = entry_times + horizon_seconds
    # Последний индекс бара, который ещё входит в горизонт
    max_future_indices = np.searchsorted(timestamps, horizon_end_times, side='right') - 1
    # Если горизонт полностью вышел за пределы массива — оставляем NaN
    has_enough_data = max_future_indices < len(close)
    active_mask = has_enough_data.copy()  # строки без данных сразу исключаем

    # Максимальное число шагов — ограничиваем чтобы не гонять пустой цикл
    max_steps = int((max_future_indices - np.arange(i_start, i_end)).max()) if active_mask.any() else 0

    for t in range(1, max_steps + 1):
        future_indices = np.arange(i_start, i_end) + t

        # Активна строка если: ещё не помечена И future не превышает её горизонт
        within_horizon = future_indices <= max_future_indices
        check_mask = active_mask & within_horizon

        exhausted = active_mask & ~within_horizon
        if np.any(exhausted):
            labels[exhausted] = 0
            active_mask[exhausted] = False

        if not np.any(check_mask):
            break  # все строки либо обработаны, либо вышли за пределы
        
        # Получаем будущие цены только для активных строк
        check_indices = np.where(check_mask)[0]
        fi = future_indices[check_indices]

        future_high = high[fi]
        future_low  = low[fi]

        # Проверяем касания границ
        hit_upper = future_high >= upper_bounds[check_indices]
        hit_lower = future_low <= lower_bounds[check_indices]
        
        # Обрабатываем одновременные касания (pos_u == pos_l)
        both_hit = hit_upper & hit_lower
        if np.any(both_hit):
            idx = check_indices[both_hit]
            labels[idx] = 2
            active_mask[idx] = False

        # Обрабатываем только верхнее касание
        only_upper = hit_upper & ~hit_lower
        if np.any(only_upper):
            idx = check_indices[only_upper]
            labels[idx] = 1
            active_mask[idx] = False

        # Обрабатываем только нижнее касание
        only_lower = hit_lower & ~hit_upper
        if np.any(only_lower):
            idx = check_indices[only_lower]
            labels[idx] = -1
            active_mask[idx] = False

    # Оставшиеся активные строки = не было касаний
    expired = active_mask & has_enough_data
    labels[expired] = 0

    return labels  # shape: (n_rows,), dtype float32

def compute_dib_thresholds(df, price_col='close', 
                            window_seconds=3600,  # rolling окно по времени
                            horizon_seconds=3600,  # горизонт лейблинга (1ч)
                            k=1.8):
    """
    Пороги для DIB через временное rolling окно.
    Scaling через реальное время бара, а не sqrt(n_bars).
    """

    log_returns = np.log(df[price_col] / df[price_col].shift(1)).rename('log_returns')

    # [1] Rolling по времени вместо баров
    rolling_std = (
        pd.concat([log_returns, df['close_time']], axis=1)
        .rolling(
            window=pd.Timedelta(seconds=window_seconds),
            min_periods=5,
            on='close_time'   # строка — имя колонки
        )['log_returns']
        .std()
    )
    
    # [2] Аннуализация через реальную длительность бара
    bar_duration = df['close_time'].diff().dt.total_seconds().bfill().rolling(window=5, min_periods=1).mean()  # сглаживаем шум длительности бара
    bar_duration = bar_duration.clip(lower=1.0)  # минимум 1 секунда чтобы не взрываться

    # Volatility scaling: σ_horizon = σ_bar * sqrt(horizon / bar_duration)
    # Для DIB bar_duration варьируется → индивидуальный scaling
    time_scale = np.sqrt(horizon_seconds / bar_duration)

    thresholds_tp = (k * rolling_std * time_scale).fillna(0).values
    thresholds_sl = (k * rolling_std * time_scale).fillna(0).values  # можно задать асимметрию

    return thresholds_tp, thresholds_sl

def load_config(config_path=None):
    """Загрузка конфигурации фич из YAML файла"""
    if config_path is None:
        # Автоматический поиск config.yaml
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent
        
        # Поиск config.yaml в родительской директории
        config_path = root_dir / "config" / "config.yaml"
        
        if not config_path.exists():
            # Альтернативные пути
            config_path = root_dir / "configs" / "config.yaml"
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def compute_event_labels_optimized(
    symbol: str, 
    timeframe: str,
    tl: int = 12,
    k: float = 1.5,  # множитель для сигмы (было threshold)
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

    # Динамические пороги для каждой строки
    horizon_seconds = (tl * pd.Timedelta(timeframe)).seconds if timeframe.find("ib") == -1 else 3600  # для imbalance_bar tl — вообще не смотрим, задаём фиксированное время
    thresholds_tp, thresholds_sl = compute_dib_thresholds(df, price_col=price_col,
                                                        window_seconds=horizon_seconds,
                                                        horizon_seconds=horizon_seconds,
                                                        k=k)

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    timestamps = df['close_time'].astype('int64').values / 1e9  # float64
    n = len(df)
    
    # Размер чанка - баланс между параллелизмом и overhead
    n_cores = n_jobs if n_jobs > 0 else 8
    chunk_size = max(1000, n // n_cores + 1)
    
    chunks = [
        (i, min(i + chunk_size, n), close, high, low, timestamps, thresholds_tp, thresholds_sl, horizon_seconds)
        for i in range(0, n, chunk_size)
    ]
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_label_chunk_optimized)(chunk) for chunk in chunks
    )
    
    labels = np.concatenate(results)
    labels[-tl:] = np.nan # Сразу мьючу последние строки
    
    labels_df = pd.DataFrame({'label': labels}, index=df.index)
    
    labels_df.to_parquet(
        processed_path,
        index=True,
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
    parser.add_argument("--config",     default="config.yaml",  help="Путь к конфигу")
    parser.add_argument("--no-args",    action="store_true",    help="Использовать значения по умолчанию без argparse")

    args = parser.parse_args()

    # Если явно попросили игнорировать аргументы → используем старый удобный пример
    if args.no_args:
        symbol      = "BTCUSDT"
        timeframe   = "dib_temp"
        config      = "config.yaml"
    else:
        symbol      = args.symbol
        timeframe   = args.timeframe
        config      = args.config

    config = load_config(CONFIG_DIR / config)  # Загружаем конфиг
    tl = config.get("labeling", {}).get("horizon_bars", 12)
    k = config.get("labeling", {}).get("threshold", 1.8)
    compute_event_labels_optimized(symbol=symbol, timeframe=timeframe, tl=tl, k=k)