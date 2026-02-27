# features/base_features.py

import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import multiprocessing as mp


# ─── CONFIG ────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=RuntimeWarning)  # для log(0) и деления на 0

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
CONFIG_DIR      = PROJECT_ROOT / "config"

def load_feature_config(config_path=None):
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
    
    return config.get('features', {})

# Использование
FEATURE_CONFIG = load_feature_config(CONFIG_DIR / "config.yaml")


# ─── HELPERS ───────────────────────────────────────────────────────────────

def true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
    """Векторизованный True Range (3)"""
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(high, low, close, window=14) -> pd.Series:
    """Classical ATR (Wilder) (2) - вопрос к alpha"""
    tr = true_range(high, low, close.shift(1))
    atr = tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return atr


def mark_invalid_rows(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Возвращает маску НЕвалидных (испорченных) строк для данной фичи с заданным lookback.
    Логика: если хотя бы один из последних lookback+1 баров был suspended → вся метрика invalid
    """
    if "is_suspended" not in df.columns:
        return pd.Series(False, index=df.index)
    
    suspended = df["is_suspended"].fillna(0).astype(bool)
    # свёртка "хотя бы один True за окно"
    invalid = suspended.rolling(lookback + 1, min_periods=1).max().astype(bool)
    # первые lookback баров всегда invalid для rolling-окон этой длины
    invalid.iloc[:lookback] = True
    return invalid


def calculate_max_window(feature_config):
    """Определяет максимальное окно из всех фич"""
    max_window = 0
    
    for feature_name, params in feature_config.items():
        if 'windows' in params:
            max_window = max(max_window, max(params['windows']))
        if 'atr_window' in params:
            max_window = max(max_window, params['atr_window'])
        if 'vol_windows' in params:
            max_window = max(max_window, max(params['vol_windows']))
        if 'zone_lookback' in params:
            max_window = max(max_window, params['zone_lookback'])
    
    return max_window


def split_with_overlap(df, n_chunks, overlap):
    """
    Разбивает DF на чанки с перекрытием (overlap) для корректного rolling
    
    Args:
        df: исходный DataFrame
        n_chunks: количество частей
        overlap: количество строк для перекрытия (макс. окно)
    
    Returns:
        list of (chunk_df, start_idx, end_idx)
        где start_idx/end_idx - индексы для финального результата
    """
    chunk_size = len(df) // n_chunks
    chunks = []
    
    for i in range(n_chunks):
        # Начало и конец "полезной" части чанка
        actual_start = i * chunk_size
        actual_end = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
        
        # Расширенные границы с overlap
        extended_start = max(0, actual_start - overlap)
        extended_end = actual_end
        
        chunk_df = df.iloc[extended_start:extended_end].copy()
        
        # Индексы для извлечения результата (относительно чанка)
        result_start = actual_start - extended_start
        result_end = result_start + (actual_end - actual_start)
        
        chunks.append((chunk_df, result_start, result_end))
    
    return chunks


def check_nan_structure(df_features, max_window):
    # первые max_window строк обязаны быть NaN
    head_nan_ratio = df_features.iloc[:max_window].isna().mean().mean()
    assert head_nan_ratio > 0.8, "Early rows should be mostly NaN"
    print(f"Initial NaN ratio (first {max_window} rows): {head_nan_ratio:.2%}")

    # после max_window NaN допустимы только из-за suspend_flag
    tail = df_features.iloc[max_window:]
    nan_ratio = tail.isna().mean().mean()
    assert nan_ratio < 0.2, "Too many NaNs after warmup — possible bug"
    print(f"Overall NaN ratio (after first {max_window} rows): {nan_ratio:.2%}")


def check_no_future_dependency(df_raw, feature_func, col_name, window_size):
    # сдвигаем исходные данные в будущее # (Не очень понятна логика, если фичу проверяем)
    shift_amount = window_size + 1  # для log_return_12 это 13

    df_shifted = df_raw.copy()
    df_shifted[["open","high","low","close","volume"]] = \
        df_shifted[["open","high","low","close","volume"]].shift(-shift_amount)

    f_orig = feature_func(df_raw)[col_name]
    f_shift = feature_func(df_shifted)[col_name]

    # Убираем NaN
    valid_idx = f_orig.notna() & f_shift.notna()
    corr = f_orig[valid_idx].corr(f_shift[valid_idx])
    print(f"Feature: {col_name}. Correlation after 1-bar shift: {corr:.4f}")
    assert abs(corr) < 0.1, "Possible future leakage detected"


# ─── FEATURE BLOCKS ────────────────────────────────────────────────────────

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    windows = FEATURE_CONFIG["returns"]["windows"]
    feats = {}
    
    for w in windows:
        feats[f"log_return_{w}"] = np.log(df["close"] / df["close"].shift(w))
    
    df_ret = pd.DataFrame(feats, index=df.index)
    
    # suspended logic
    invalid_mask = mark_invalid_rows(df, max(windows))
    df_ret[invalid_mask] = np.nan
    
    return df_ret.add_prefix("return__")


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    atr_w = FEATURE_CONFIG["volatility"]["atr_window"]
    vol_ws = FEATURE_CONFIG["volatility"]["vol_windows"]
    percentile_ws = FEATURE_CONFIG["volatility"]["percentile_windows"]
    
    atr = compute_atr(df["high"], df["low"], df["close"], atr_w)
    norm_atr = atr / df["close"]
    
    logret = np.log(df["close"] / df["close"].shift(1))
    vol_feats = {}
    vol_feats["atr_14"] = atr
    vol_feats["norm_atr_14"] = norm_atr
    
    for w in vol_ws:
        vol_feats[f"vol_logret_{w}"] = logret.rolling(w).std()
    
    # (A) Volatility regime positioning
    vol_20 = logret.rolling(20).std()
    for w in percentile_ws:
        # vol_feats[f"vol_percentile_{w}"] = vol_20.rolling(w).apply(
        #     lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        # )
        vol_feats[f"vol_percentile_{w}"] = vol_20.rolling(w).rank(pct=True)

    # (B) Volatility slope / expansion
    vol_10 = logret.rolling(10).std()
    vol_feats["vol_expansion_ratio"] = vol_10 / (vol_20 + 1e-8)  # ratio
    vol_feats["vol_expansion_diff"] = vol_10 - vol_20  # difference
    
    # (C) Volatility of volatility
    vol_feats["vol_of_vol_20"] = vol_20.rolling(20).std()
    
    df_vol = pd.DataFrame(vol_feats, index=df.index)
    
    max_window = max(atr_w, max(vol_ws), max(percentile_ws))
    invalid_mask = mark_invalid_rows(df, max_window)
    df_vol[invalid_mask] = np.nan
    
    return df_vol.add_prefix("volatility__")


def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    windows = FEATURE_CONFIG["range"]["windows"]
    feats = {}
    
    for w in windows:
        rng_mean = (df["high"] - df["low"]).rolling(w).mean()
        feats[f"range_mean_{w}"] = rng_mean / df["close"]
        
    feats["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-8)
    
    df_rng = pd.DataFrame(feats, index=df.index)
    
    invalid_mask = mark_invalid_rows(df, max(windows))
    df_rng[invalid_mask] = np.nan
    
    return df_rng.add_prefix("range__")


def compute_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    lb = FEATURE_CONFIG["liquidity"]["zone_lookback"]
    
    rolling_max = df["high"].rolling(lb).max()
    rolling_min = df["low"].rolling(lb).min()
    
    dist_high = (df["close"] - rolling_max) / rolling_max
    dist_low  = (df["close"] - rolling_min) / rolling_min
    position  = (df["close"] - rolling_min) / (rolling_max - rolling_min)
    
    df_liq = pd.DataFrame({
        "dist_to_high": dist_high,
        "dist_to_low":  dist_low,
        "position_in_range": position
    }, index=df.index)
    
    invalid_mask = mark_invalid_rows(df, lb)
    df_liq[invalid_mask] = np.nan
    
    return df_liq.add_prefix("liquidity__")


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    windows = FEATURE_CONFIG["volume"]["windows"]
    feats = {}
    
    for w in windows:
        mean_vol = df["volume"].rolling(w).mean()
        std_vol  = df["volume"].rolling(w).std()
        feats[f"volume_z_{w}"] = (df["volume"] - mean_vol) / std_vol
        
        dollar_vol = df["volume"] * df["close"]
        mean_dv = dollar_vol.rolling(w).mean()
        std_dv  = dollar_vol.rolling(w).std()        
        feats[f"dollar_volume_z_{w}"] = (dollar_vol - mean_dv) / std_dv
    
    df_vol = pd.DataFrame(feats, index=df.index)
    
    invalid_mask = mark_invalid_rows(df, max(windows))
    df_vol[invalid_mask] = np.nan
    
    return df_vol.add_prefix("volume__")


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:    
    dt = df.index if df.index.name == "open_time" else pd.to_datetime(df["open_time"])
    
    hour = dt.hour + dt.minute / 60
    dow  = dt.dayofweek + dt.hour / 24
    
    feats = {}
    if FEATURE_CONFIG["time"]["hour_sin_cos"]:
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    if FEATURE_CONFIG["time"]["day_of_week"]:
        feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    if not feats:
        return pd.DataFrame(index=df.index)
    
    return pd.DataFrame(feats, index=df.index).add_prefix("time__")


def compute_realized_skewness(df: pd.DataFrame) -> pd.DataFrame:
    windows = FEATURE_CONFIG["realized_skewness"]["windows"]
    
    logret = np.log(df["close"] / df["close"].shift(1))
    
    feats = {}
    for w in windows:
        feats[f"realized_skew_{w}"] = logret.rolling(w).skew()
    
    # Дополнительно: intrabar directional movement (если есть OHLC)
    intrabar_range = df["high"] - df["low"]
    close_position = (df["close"] - df["low"]) / (intrabar_range + 1e-8)  # 0 = close at low, 1 = close at high
    
    for w in windows:
        feats[f"close_position_mean_{w}"] = close_position.rolling(w).mean()
    
    df_skew = pd.DataFrame(feats, index=df.index)
    
    max_window = max(windows)
    invalid_mask = mark_invalid_rows(df, max_window)
    df_skew[invalid_mask] = np.nan
    
    return df_skew.add_prefix("skewness__")


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

FEATURE_COMPUTERS = [
    compute_returns,
    compute_volatility,
    compute_range_features,
    compute_liquidity_features,
    compute_volume_features,
    compute_time_features,
    compute_realized_skewness,  # <- новая функция
]


def compute_all_features_chunk(args) -> pd.DataFrame:
# def compute_all_features_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """Вычисляем все группы фичей для одного куска данных с учётом overlap"""
    df_chunk, start_idx, end_idx = args
    dfs = []
    for func in FEATURE_COMPUTERS:
        try:
            f = func(df_chunk)
            # Возвращаем только "полезную" часть без overlap
            dfs.append(f.iloc[start_idx:end_idx])
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(index=df_chunk.index)
    
    return pd.concat(dfs, axis=1)


def parallel_compute(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    # 1. Вычисляем максимальное окно
    max_window = calculate_max_window(FEATURE_CONFIG)
    print(f"Max window detected: {max_window}")
    
    # 2. Разбиваем с overlap
    chunks = split_with_overlap(df, n_chunks=n_jobs, overlap=max_window)
    
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(compute_all_features_chunk, chunks)
        
    all_results = pd.concat(results)
    all_results[:max_window] = np.nan  # зануляем первые строки

    check_nan_structure(all_results, max_window)

    # Дополнительная проверка на утечку будущего. Можно включить по необходимости
    for feature_func, col_name, window_size in [
        (compute_returns, "return__log_return_1", 1),
        (compute_volume_features, "volume__volume_z_20", 20),
        # (compute_volatility, "volatility__norm_atr_14", 14),          # market persistent
        # (compute_range_features, "range__range_mean_10", 10),         # market persistent
        # (compute_liquidity_features, "liquidity__dist_to_high", 200), # trend affected
    ]:
        check_no_future_dependency(df, feature_func, col_name, window_size)
    
    return all_results