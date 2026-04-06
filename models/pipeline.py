# models/pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.build_dataset import build_dataset
from dataset.splits import time_split
from models.baseline import BarrierEstimator, BaselineClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from evaluation.strategies import (
    BaselineStrategy,
    TwoStageStrategy,
    EVStrategy
)
from evaluation.comparison import StrategyComparison, compare_strategies_grid

from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardAnalysis
from backtest.metrics import calculate_performance_metrics, print_performance_summary



INTERIM_DIR     = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR      = PROJECT_ROOT / "config"


# ─── HELPERS ───────────────────────────────────────────────────────────────

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


def compute_trade_metrics(y_true, y_pred):
    """ 
    Precision = правильные_направления / все_сделки
    
    Считаем только для ненулевых предсказаний (реальных трейдов)
    """
    trade_mask = y_pred != 0
    
    if trade_mask.sum() == 0:
        return 0.0
    
    y_true_trades = y_true[trade_mask]
    y_pred_trades = y_pred[trade_mask]
    
    # Правильное направление = знаки совпали
    correct_direction = (y_true_trades * y_pred_trades) > 0
    metrics = {
        "precision": correct_direction.sum() / len(y_pred_trades),
        "trade_freq": trade_mask.mean(),
    }
    
    return metrics


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def run_pipeline(
    symbol: str,
    timeframe: str,
    config: str,
) -> pd.DataFrame:
    
    symbol = symbol.upper()
    path_interim    = INTERIM_DIR / f"{symbol}_{timeframe}.parquet"
    path_features   = PROCESSED_DIR / f"{symbol}_{timeframe}_features.parquet"
    path_labels     = PROCESSED_DIR / f"{symbol}_{timeframe}_labels.parquet"

    df_interim = pd.read_parquet(path_interim)
    features = pd.read_parquet(path_features)
    labels = pd.read_parquet(path_labels)

    print(f"Loaded interim data: {df_interim.shape}, features: {features.shape}, labels: {labels.shape}")
    config_file  = load_config(CONFIG_DIR / config)
    feature_config = config_file['features']
    max_lookback    = calculate_max_window(feature_config)
    
    df = build_dataset(features, labels['label'], df_interim["is_suspended"], max_lookback=max_lookback)
    
    # split
    print(f"Splitting dataset by time...")
    val_start_date = config_file['training']['val_start']
    test_start_date = config_file['training']['test_start']
    time_periods = time_split(df, time_periods=[val_start_date, test_start_date])#"2025-01-01")
    if len(time_periods) != 3:
        raise ValueError(f"Expected 3 time periods (train, val, test), but got {len(time_periods)}. Check your time_periods configuration.")
    train, val, test = time_periods

    print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} samples)")
    print(f"Val: {val.index.min()} to {val.index.max()} ({len(val)} samples)")
    print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} samples)")

    X_train, y_train = train.drop(columns="label"), train["label"]
    X_val, y_val = val.drop(columns="label"), val["label"]
    X_test, y_test = test.drop(columns="label"), test["label"]
    
    # Подтягиваю настройки модели из config
    config_model_dict = {
        "lgbm_regression": LGBMClassifier,
        "XGBC_regression": XGBClassifier,
        "logistic_regression": LogisticRegression
    }
    config_model = config_file['model']
    if config_model['type'] not in config_model_dict.keys():
        raise(f"Invalid model type {config_model['type']}. Possible options: {' / '.join(config_model_dict.keys())}")
    estimator = config_model_dict[config_model['type']](
        **config_model['params']
    )

    # Расчет границ для последующих оценок EV стратегий
    com_rate = (config_file['backtest']['commission_bps'] + config_file['backtest']['slippage_bps']) / 10000
    barrier_configs = {
        "atr_window": config_file['labeling']['horizon_bars'],
        "k": config_file['labeling']['threshold'],
        "com_rate": com_rate,
    } # slippage + commission из расчета, что объем позиции почти не двигает цену
    barriers = BarrierEstimator(**barrier_configs).fit(df_interim)
    
    # ================== WALK-FORWARD VALIDATION ==================
    
    print(f"\n{'='*80}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*80}\n")
    
    # Создаем стратегию для WF (обнуление моделей внутри WFA)
    wf_models = {
        "model_hit": BaselineClassifier(estimator=estimator, calibrate=True, calibration_method="isotonic"),
        "model_direction": BaselineClassifier(estimator=estimator, calibrate=True, calibration_method="isotonic"),
        "model_baseline": BaselineClassifier(estimator=estimator, calibrate=True, calibration_method="isotonic"),
    }
    
    wf_validator = WalkForwardAnalysis(
        models=wf_models,
        com_rate=com_rate,
        initial_capital=config_file['backtest']['initial_capital'],
        mdd=config_file['risk']['max_drawdown'],
    )
    
    # Split схема
    parts = [0.6, 0.2, 0.2]  # Train 60%, Val 20%, Test 20%
    start_end_wfa_splits = [
        [0.0, 0.8],  # start 0%, end 80% для 1 шага WFA
        [0.2, 1.0],  # start 0%, end 100% для 2 шага WFA
    ]
    splits = [tuple(parts + a_b) for a_b in start_end_wfa_splits] # комбинируем одинаковое разбиение с разными start/end точками для WFA
    
    # Набор стратегий
    Baseline_Strategies = {
        "name": "Baseline",
        "model": BaselineStrategy,
        "required_models": ["model_baseline"],
        "hyperparams": {
            "threshold": [0.35, 0.4, 0.5, 0.6, 0.65, 0.7],
        }
    }
    TwoStage_Strategies = {
        "name": "TwoStage",
        "model": TwoStageStrategy,
        "required_models": ["model_hit", "model_direction"],
        "hyperparams": {
            "hit_threshold": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "direction_threshold": [0.5, 0.6],
        }
    }
    EV_Strategies = {
        "name": "EV",
        "model": EVStrategy,
        "required_models": ["model_hit", "model_direction"],
        "hyperparams": {
            "min_ev": [0.0, 0.001, 0.002],
            "min_probability": [0.0, 0.05, 0.1, 0.15, 0.20, 0.25],
        }
    }
    strategies = {
        "Baseline": Baseline_Strategies,
        "TwoStage": TwoStage_Strategies,
        "EV": EV_Strategies,
    }
    
    # Запускаем WF
    # ВАЖНО: нужно передать всю выборку (train + val + test)
    wf_results = wf_validator.run(
        strategies=strategies,
        X=pd.concat([X_train, X_val, X_test]),
        y=pd.concat([y_train, y_val, y_test]),
        df_ohlc=df_interim.loc[df.index],
        splits=splits,
        barriers=barriers.loc[df.index],
    )

# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",     default="BTCUSDT",     help="Торговая пара")
    parser.add_argument("--timeframe",  default="5m",          help="Таймфрейм")
    parser.add_argument("--config",     default="config.yaml", help="Конфигурация фич")
    parser.add_argument("--no-args",    action="store_true",   help="Использовать значения по умолчанию без argparse")

    
    args = parser.parse_args()

    # Если явно попросили игнорировать аргументы → используем старый удобный пример
    if args.no_args:
        symbol      = "BTCUSDT"
        timeframe   = "5m"
        config      = "config.yaml"
    else:
        symbol      = args.symbol
        timeframe   = args.timeframe
        config      = args.config

    pipe_results = run_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        config=config
    )
    print("\nPipeline execution completed.")