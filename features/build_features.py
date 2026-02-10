# features/build_features.py

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from .base_features import parallel_compute
from .feature_cleaning import clean_features


PROJECT_ROOT    = Path(__file__).resolve().parents[1]
INTERIM_DIR     = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR      = PROJECT_ROOT / "config"

# ─── CONFIG ────────────────────────────────────────────────────────────────

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
        path_in     = INTERIM_DIR / "BTCUSDT_5m.parquet"
        path_out    = PROCESSED_DIR / "BTCUSDT_5m_features.parquet"
        labels      = PROCESSED_DIR / "BTCUSDT_5m_labels.parquet"
        n_jobs      = -1
    else:
        path_in     = Path(args.input)
        path_out    = Path(args.output)
        labels      = Path(args.labels)
        n_jobs      = args.n_jobs

    path_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {path_in} ...")
    df = pd.read_parquet(path_in)
    
    # предполагаем, что индекс — open_time (datetime)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "open_time" in df.columns:
            df = df.set_index("open_time")
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError("Ожидается datetime index или колонка 'open_time'")

    df = df.sort_index()

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