# validation/check_features.py

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path


# ─── CONFIG ────────────────────────────────────────────────────────────────

logger = logging.getLogger("check_features")

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
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

# Загрузка данных с конфига
FEATURE_CONFIG = load_feature_config(CONFIG_DIR / "config.yaml")


# ─── HELPERS ───────────────────────────────────────────────────────────────

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

def check_nan_structure(df_features, max_window):
    # первые max_window строк обязаны быть NaN
    head_nan_ratio = df_features.iloc[:max_window].isna().mean().mean()
    assert head_nan_ratio > 0.8, "Early rows should be mostly NaN"

    # после max_window NaN допустимы только из-за suspend_flag
    tail = df_features.iloc[max_window:]
    nan_ratio = tail.isna().mean().mean()
    assert nan_ratio < 0.2, "Too many NaNs after warmup — possible bug"
    return True
    

def check_feature_distributions(df):
    stats = df.describe(percentiles=[0.01, 0.99]).T

    # бесконечности
    assert not np.isinf(stats.values).any(), "Inf values detected"

    # слишком узкое распределение
    near_constant = stats["std"] < 1e-6
    assert near_constant.sum() == 0, f"Near-constant features: {stats[near_constant].index.tolist()}"
    return True
    

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def check_features(path: Path):
    success = True  
    if not path.is_file():
        logger.error(f"Файл не найден: {path}")
        return

    try:
        df = pd.read_parquet(path)
        logger.info(f"Прочитано {len(df):,} строк")
    except Exception as e:
        logger.exception(f"Ошибка чтения parquet: {e}")
        return

    max_window = calculate_max_window(FEATURE_CONFIG)
    
    success = all([success, check_nan_structure(df, max_window)])
    success = all([success, check_feature_distributions(df)])
    
    if success:
        logger.info("Валидация пройдена Успешно")
    else:
        logger.error("Валидация НЕ пройдена → файл НЕ сохранён")


# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="BTCUSDT_5m_features.parquet",  help="путь к .parquet с features")
    parser.add_argument("--no-args",    action="store_true",                    help="Использовать значения по умолчанию без argparse")

    args = parser.parse_args()

    # Если явно попросили игнорировать аргументы → используем старый удобный пример
    if args.no_args:
        path    = PROCESSED_DIR / "BTCUSDT_5m_features.parquet"
    else:
        path    = Path(args.input_path)

    check_features(path=path)