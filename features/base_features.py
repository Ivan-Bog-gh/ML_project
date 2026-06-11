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
    expansion_ws = FEATURE_CONFIG["volatility"]["expansion_windows"]
    vol_of_vol_ws = FEATURE_CONFIG["volatility"]["vol_of_vol_windows"]
    
    calculations = pd.DataFrame()
    calculations['atr'] = compute_atr(df["high"], df["low"], df["close"], atr_w)
    calculations['norm_atr'] = calculations['atr'] / df["close"]
    
    calculations['logret'] = np.log(df["close"] / df["close"].shift(1))
    vol_feats = {}
    vol_feats["atr_14"] = calculations['atr']
    vol_feats["norm_atr_14"] = calculations['norm_atr']
    
    for w in vol_ws:
        vol_feats[f"vol_logret_{w}"] = calculations['logret'].rolling(w).std()
    
    # (A) Volatility regime positioning
    calculations['vol_20'] = calculations['logret'].rolling(20).std()
    for w in percentile_ws:
        vol_feats[f"vol_percentile_{w}"] = calculations['vol_20'].rolling(w).rank(pct=True)

    # (B) Volatility slope / expansion
    calculations['vol_10'] = calculations['logret'].rolling(10).std()
    for short_ws, long_ws in expansion_ws:
        vol_short = f"vol_{short_ws}"
        vol_long = f"vol_{long_ws}"

        if vol_short not in calculations.columns:
            calculations[vol_short] = calculations['logret'].rolling(short_ws).std()
            
        if vol_long not in calculations.columns:
            calculations[vol_long] = calculations['logret'].rolling(long_ws).std()

        vol_feats[f"vol_expansion_ratio_{short_ws}_{long_ws}"] = calculations[vol_short] / (calculations[vol_long] + 1e-8)  # ratio
        vol_feats[f"vol_expansion_diff_{short_ws}_{long_ws}"] = calculations[vol_short] - calculations[vol_long]  # difference
    
    # (C) Volatility of volatility
    for w_1, w_2 in vol_of_vol_ws:
        vol_w_1 = f"vol_{w_1}"
        if vol_w_1 not in calculations.columns:
            calculations[vol_w_1] = calculations['logret'].rolling(w_1).std()

        vol_feats[f"vol_{w_2}_of_vol_{w_1}"] = calculations[vol_w_1].rolling(w_2).std()
    
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
    intrabar_windows = FEATURE_CONFIG["realized_skewness"]["intrabar_direction_windows"] 
    
    logret = np.log(df["close"] / df["close"].shift(1))
    
    feats = {}
    for w in windows:
        feats[f"realized_skew_{w}"] = logret.rolling(w).skew()
    
    # Дополнительно: intrabar directional movement
    close_position = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)  # 0 = close at low, 1 = close at high
    
    for w in intrabar_windows:
        feats[f"close_position_mean_{w}"] = close_position.rolling(w).mean()
    
    df_skew = pd.DataFrame(feats, index=df.index)
    
    max_window = max(windows)
    invalid_mask = mark_invalid_rows(df, max_window)
    df_skew[invalid_mask] = np.nan
    
    return df_skew.add_prefix("skewness__")


def compute_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    # Кумулятивная body_ratio. less -> short
    signed_body = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)
    feats = {}
    for w in [3, 6, 12]:
        feats[f'signed_body_sum_{w}'] = signed_body.rolling(w).sum()

    # 2. Price vs SMA — отклонение со знаком
    for w in [20, 50]:
        ma = df["close"].rolling(w).mean()
        feats[f'price_vs_sma_{w}'] = (df["close"] - ma) / (df["close"].rolling(w).std() + 1e-8)

    # 3. RSI нормализованный
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    feats['rsi_14'] = (100 - (100 / (1 + up / down)) - 50) / 50  # [-1, +1]

    # 4. MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    feats['macd'] = (ema12 - ema26) / df["close"]
    
    df_dir = pd.DataFrame(feats, index=df.index)
    
    invalid_mask = mark_invalid_rows(df, 50)  # максимум из windows
    df_dir[invalid_mask] = np.nan
    
    return df_dir.add_prefix("direction__")


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    # windows = FEATURE_CONFIG["realized_skewness"]["windows"]

    # ---- Вспомогательные ряды ----
    logret = np.log(df["close"] / df["close"].shift(1))
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = df['high'] - df['low']
    # df['atr_14'] = df['high_low_range'].rolling(14).mean()
    atr = compute_atr(df["high"], df["low"], df["close"], 14)
    norm_atr = atr / df["close"]
    
    feats = {}
    
    # ---- 1. Признаки CVD (поток агрессивных ордеров) ----
    # Изменение CVD за 1 час (12 баров)
    feats['cvd_delta_1h'] = df['cvd'].diff(12)
    # ?? Нормализованное изменение CVD (в единицах ATR) -> почему $ делим на range_size
    feats['cvd_norm_delta_1h'] = feats['cvd_delta_1h'] / atr.replace(0, np.nan)
    # Ускорение CVD (вторая разность)
    feats['cvd_accel_1h'] = df['cvd'] - 2*df['cvd'].shift(6) + df['cvd'].shift(12)
    # Наклон CVD за 24 бара (2 часа) – быстрая аппроксимация через разность средних
    feats['cvd_slope_2h'] = (df['cvd'].rolling(24).mean() - 
                          df['cvd'].shift(24).rolling(24).mean()) / 24
    
    # ---- 2. Признаки OI Proxy (открытый интерес) ----
    feats['oi_proxy_delta_1h'] = df['oi_proxy'].diff(12)
    feats['oi_proxy_roc_1h'] = df['oi_proxy'].pct_change(12)
    # Соотношение текущего OI к его 12-часовой средней
    feats['oi_proxy_ma_ratio'] = df['oi_proxy'] / df['oi_proxy'].rolling(144).mean()
    # Корреляция OI и цены (24 бара)
    feats['oi_price_corr_24'] = df['oi_proxy'].rolling(24).corr(df['close'])
    
    # ---- 3. Объём и торговая активность ----
    # Соотношение объёма к среднему за 12 часа
    feats['volume_ratio_12h'] = df['volume'] / df['volume'].rolling(144).mean()
    # Соотношение числа сделок к среднему за 12 часа
    feats['trades_count_ratio_12h'] = df['trades_count'] / df['trades_count'].rolling(144).mean()
    # Нормализованный средний размер сделки (z-score за 12 часа)
    avg_trade_size_mean = df['avg_trade_size'].rolling(144).mean()
    avg_trade_size_std = df['avg_trade_size'].rolling(144).std()
    feats['avg_trade_size_zscore'] = (df['avg_trade_size'] - avg_trade_size_mean) / avg_trade_size_std.replace(0, np.nan)
    # Соотношение агрессивных покупок к продажам (используем volume_delta для аппроксимации)
    feats['buy_sell_ratio_1h'] = ((df['volume'] + df['volume_delta']) / 
                                   (df['volume'] - df['volume_delta'] + 1e-9)).rolling(12).mean()
    
    # ---- 4. Волатильность и режим рынка (дополнение к вашим) ----
    # Bollinger Bands %b
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    feats['bb_pct_b'] = (df['close'] - (bb_mid - 2*bb_std)) / (4 * bb_std + 1e-9)
    # Ширина полос Боллинджера (нормализованная)
    feats['bb_width'] = (4 * bb_std) / bb_mid
    # Процентиль ширины BB за 12 часа (индикатор сжатия/расширения)
    feats['bb_width_percentile'] = feats['bb_width'].rolling(144).rank(pct=True)
    # Соотношение короткой и длинной ATR (импульс волатильности)
    # df['atr_ratio_5_50'] = df['atr_14'] / df['high_low_range'].rolling(50).mean()
    
    # ---- 5. Эффективность движения (шум vs тренд) ----
    net_change = (df['close'] - df['close'].shift(6)).abs()
    sum_abs_change = (df['close'] - df['close'].shift(1)).abs().rolling(6).sum()
    feats['efficiency_ratio'] = net_change / (sum_abs_change + 1e-9)
    
    # ---- 6. Взаимодействие цены и потока (CVD) ----
    feats['price_cvd_corr_24'] = df['close'].rolling(24).corr(df['cvd'])
    # Дивергенция: растёт цена, но CVD падает
    feats['cvd_price_divergence'] = np.sign(df['close'].diff(12)) != np.sign(feats['cvd_delta_1h'])
    feats['cvd_price_divergence'] = feats['cvd_price_divergence'].astype(float)
    
    # ---- 7. Скорость изменения среднего размера сделки (институциональный след) ----
    feats['avg_trade_size_mom'] = df['avg_trade_size'] / df['avg_trade_size'].shift(6).replace(0, np.nan) - 1
    
    # ---- 8. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ----
    
    # 8a. Отклонение от VWAP (12 часа), нормализованное ATR
    vwap_num = (df['close'] * df['volume']).rolling(144).sum()
    vwap_den = df['volume'].rolling(144).sum()
    vwap = vwap_num / vwap_den.replace(0, np.nan)
    feats['vwap_deviation'] = (df['close'] - vwap) / atr.replace(0, np.nan)
    
    # 8b. Истощение CVD: когда CVD достиг пика за последние 24 бара и начал снижаться
    rolling_max_cvd = df['cvd'].rolling(24).max()
    rolling_min_cvd = df['cvd'].rolling(24).min()
    # Признак: насколько текущий CVD близок к максимуму и при этом delta отрицательна
    feats['cvd_exhaustion'] = ((df['cvd'] >= rolling_max_cvd * 0.98) & (feats['cvd_delta_1h'] < 0)) | \
                           ((df['cvd'] <= rolling_min_cvd * 1.02) & (feats['cvd_delta_1h'] > 0))
    feats['cvd_exhaustion'] = feats['cvd_exhaustion'].astype(float)
    
    # 8c. Флаг крупных сделок: доля объёма крупных сделок относительно общего объёма за последние 144 баров
    # Используем порог 95-го процентиля avg_trade_size в скользящем окне 144 баров
    threshold_95 = df['avg_trade_size'].rolling(144).quantile(0.95)
    # Для каждой строки определяем, превышает ли avg_trade_size порог (если да, считаем объём как крупный)
    is_large = df['avg_trade_size'] > threshold_95
    # Сумма объёма крупных сделок за последние 144 баров
    large_volume_sum = (df['volume'] * is_large).rolling(144).sum()
    total_volume_sum = df['volume'].rolling(144).sum()
    feats['large_trade_volume_ratio'] = large_volume_sum / total_volume_sum.replace(0, np.nan)
    # Также можно сделать долю числа крупных сделок
    large_trade_count_sum = (is_large).rolling(144).sum()
    total_trade_count_sum = df['trades_count'].rolling(144).sum()
    feats['large_trade_count_ratio'] = large_trade_count_sum / total_trade_count_sum.replace(0, np.nan)

    # # Удаляем временные колонки, оставляем только новые признаки
    # cols_to_drop = ['returns', 'high_low_range']
    # df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    df_micro = pd.DataFrame(feats, index=df.index)
    
    invalid_mask = mark_invalid_rows(df, 200)
    df_micro[invalid_mask] = np.nan
    
    return df_micro.add_prefix("microstructure__")


def compute_dibs_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['dollar_imbalance'] = 2 * df['taker_buy_base'] - df['volume']

    feats = {}
    feats['dollar_imbalance_abs'] = abs(df['dollar_imbalance'])
    feats['imbalance_intensity'] = feats['dollar_imbalance_abs'] / (df['volume'] + 1e-8)  # Чем выше, тем более "однобоким" был бар
    feats['net_buying_pressure'] = df['dollar_imbalance'] / (df['volume'] + 1e-8) # Дисбаланс в % от общего объёма (-1 -> 1)
    feats['imbalance_volatility'] = df['dollar_imbalance'].rolling(20).std()
    feats['imbalance_volatility_ratio'] = (
        feats['dollar_imbalance_abs'] / (feats['imbalance_volatility'] + 1e-8)
    )
    feats['price_impact_per_dollar'] = (df['close'] - df['open']) / (feats['dollar_imbalance_abs'] + 1e-8)
    sig_price = np.where(df['dollar_imbalance'] < 0, df['low'], df['high'])  # если продавцы доминируют — смотрим на low, иначе на high
    feats['wap'] = feats['net_buying_pressure'] * abs(sig_price - df['open']) + df['open'] # Weighted Average Price impact (-1 -> low / 1 -> high)
    feats['wap'] = 1 - feats['wap'] / df['close']  # Сравнение ожидаемой цены (с учетом buying pressure) с фактической. >0 -> покупатели сильные?
    feats['trades_cnt'] = df['trades_count']
    if 'open_time' in df.columns:
        feats['time_s'] = (df['close_time'] - df['open_time']).dt.seconds  # для оценки скорости дисбаланса
    else:
        feats['time_s'] = (df['close_time'] - df.index).dt.seconds
    feats['trades_speed'] = df['trades_count'] / (feats['time_s'] + 1e-8)  # сделок в секунду
    feats['volume_speed'] = df['volume'] / (feats['time_s'] + 1e-8)  # объем в секунду
    for w in [5, 20]:
        feats[f'cum_imbalance_{w}'] = df['dollar_imbalance'].rolling(w).sum()
        feats[f'net_buying_pressure_{w}'] = feats['net_buying_pressure'].rolling(w).sum()
        feats[f'wap_{w}'] = feats['wap'].rolling(w).mean()  # Сглаженный WAP/close для оценки устойчивости давления
        feats[f'trades_cnt_ratio_{w}'] = df['trades_count'] / df['trades_count'].rolling(w).mean() # С уменьшением - ускорение торгового дисбаланса
        feats[f'time_s_ratio_{w}'] = feats['time_s'] / feats['time_s'].rolling(w).mean() # С уменьшением - ускорение торгового дисбаланса
        feats[f'volume_speed_ratio_{w}'] = feats['volume_speed'] / feats['volume_speed'].rolling(w).mean() # С увеличением - ускорение торгового дисбаланса
        feats[f'trades_speed_ratio_{w}'] = feats['trades_speed'] / feats['trades_speed'].rolling(w).mean() # С увеличением - ускорение торгового дисбаланса
    
    # Если после 3 баров с положительным дисбалансом идет отрицательный — разворот
    feats['imbalance_signal'] = np.where(
        (df['dollar_imbalance'] > 0) & 
        (df['dollar_imbalance'].shift(1) < 0) &
        (df['dollar_imbalance'].shift(2) < 0), 
        1,
        np.where(
        (df['dollar_imbalance'] < 0) & 
        (df['dollar_imbalance'].shift(1) > 0) &
        (df['dollar_imbalance'].shift(2) > 0), -1, 0)  # потенциальный разворот вниз
    )

    df_dibs = pd.DataFrame(feats, index=df.index)
    max_window = max(20, 5)  # максимум из окон для rolling
    invalid_mask = mark_invalid_rows(df, max_window)
    df_dibs[invalid_mask] = np.nan
    
    return df_dibs.add_prefix("DIBs__")

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

FEATURE_COMPUTERS = [
    compute_returns,
    compute_volatility,
    compute_range_features,
    compute_liquidity_features,
    compute_volume_features,
    compute_time_features,
    compute_realized_skewness,
    compute_direction_features,
    compute_microstructure_features,
    compute_dibs_features,
]


def compute_all_features_chunk(args) -> pd.DataFrame:
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