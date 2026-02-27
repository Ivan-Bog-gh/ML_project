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
from models.baseline import BaselineClassifier, ExpectedValueTrader

from evaluation.strategies import (
    BaselineStrategy,
    TwoStageStrategy,
    EVStrategy
)
from evaluation.comparison import StrategyComparison, compare_strategies_grid



INTERIM_DIR     = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR      = PROJECT_ROOT / "config"


# ─── HELPERS ───────────────────────────────────────────────────────────────

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


def getTEvents(gRaw, h):
    '''
        Применение CUSUM-метода для отлавливания ситуаций, когда цена вышла из проторговки
        gRaw = ряд метрик (цена/индикаторы/прочее), которые смотрим на уход из проторговки
        tEvents = ряд timestamp-ов, когда произошло событие для включения дальнейшей стратегии (Мета-маркировка)
    '''
    tEvents, sPos, sNeg = [], 1, 1
    diff = gRaw.diff() / gRaw.shift(1) # относительное изменение цены
    for i in diff.index[1:]:
        sPos, sNeg = max(1,sPos * (1 + diff.loc[i])), min(1, sNeg * (1 + diff.loc[i]))
        if sNeg < 1 - h:
            sNeg = 1
            tEvents.append(i)
        elif sPos > 1 + h:
            sPos = 1
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

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
    # path_events     = PROCESSED_DIR / f"{symbol}_{timeframe}_events_0.01.parquet"

    df_interim = pd.read_parquet(path_interim)
    features = pd.read_parquet(path_features)
    labels = pd.read_parquet(path_labels)
    # events = pd.read_parquet(path_events)
    # print(events.head())

    print(f"Loaded interim data: {df_interim.shape}, features: {features.shape}, labels: {labels.shape}")
    feature_config  = load_feature_config(CONFIG_DIR / config)
    max_lookback    = calculate_max_window(feature_config)
    
    df = build_dataset(features, labels['label'], df_interim["is_suspended"], max_lookback=max_lookback)
 
    # print(f"Dataset before applying CUSUM filter: {df.shape}")
    # df = df[df.index.isin(events.index)]
    # print(f"Dataset after applying CUSUM filter: {df.shape}")
    
    # split
    print(f"Splitting dataset by time...")
    train, val = time_split(df, split_date="2025-01-01")

    X_train, y_train = train.drop(columns="label"), train["label"]
    X_val, y_val = val.drop(columns="label"), val["label"] # ~2,500 samples

    # create binary classification: hit OR no hit
    y_train_hit = y_train.replace(-1, 1) # 0 = no-hit, 1 = hit (long or short)
    y_val_hit = y_val.replace(-1, 1)
    print(f"y_train_hit value counts (normalized):")
    print(y_train_hit.value_counts(normalize=True))
    print(f"y_val_hit value counts (normalized):")
    print(y_val_hit.value_counts())
    
    # train hit classifier (hit vs no hit)
    print(f"Training hit classifier...")
    clf_hit = BaselineClassifier(calibrate=True, calibration_method="isotonic")
    clf_hit.fit(X_train, y_train_hit)
    clf_hit_metrics = clf_hit.evaluate_roc_auc(X_val, y_val_hit)
    print(f"Hit classifier ROC AUC: {clf_hit_metrics['roc_auc']:.4f}")

    # create dataset for directional classifier (только hit)
    bin_mask_train = y_train != 0
    bin_mask_val = y_val != 0
    X_train_bin, y_train_bin = X_train[bin_mask_train], y_train[bin_mask_train]
    X_val_bin, y_val_bin = X_val[bin_mask_val], y_val[bin_mask_val]
    print(f"y_train_bin value counts (normalized):")
    print(y_train_bin.value_counts(normalize=True))
    print(f"y_val_bin value counts (normalized):")
    print(y_val_bin.value_counts())

    # train Directional classifier
    print(f"Training Directional classifier...")
    clf_direction = BaselineClassifier(calibrate=True, calibration_method="isotonic")
    clf_direction.fit(X_train_bin, y_train_bin)
    clf_direction_metrics = clf_direction.evaluate_roc_auc(X_val_bin, y_val_bin)
    print(f"Directional classifier ROC AUC: {clf_direction_metrics['roc_auc']:.4f}")

    # train all-in-one baseline
    print(f"\nTraining baseline classifier...")
    clf_baseline = BaselineClassifier()
    clf_baseline.fit(X_train, y_train)
    baseline_metrics = clf_baseline.evaluate(X_val, y_val, threshold=0.5)
    print(f"Baseline Precision: {baseline_metrics['precision']:.4f}")
    print(f"Baseline Trade Freq: {baseline_metrics['trade_freq']:.4f}")

    # ================== СОЗДАНИЕ СТРАТЕГИЙ ==================
    
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}\n")
    
    comparison = StrategyComparison()
    
    # Стратегия 1: Baseline с разными thresholds
    print("Creating Baseline strategies...")
    for thresh in [0.35, 0.4, 0.5, 0.6, 0.65, 0.7]:
        strategy = BaselineStrategy(
            name=f"Baseline_thresh{thresh:.2f}",
            model=clf_baseline,
            threshold=thresh,
        )
        comparison.evaluate_strategy(strategy, X_val, y_val)
    
    # Стратегия 2: Two-Stage с grid search
    print("\nCreating TwoStage strategies...")
    for hit_thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:  # можно варьировать threshold для hit-модели
        for dir_thresh in [0.5]: # [0.5, 0.6, 0.7]
            strategy = TwoStageStrategy(
                name=f"TwoStage_h{hit_thresh:.2f}_d{dir_thresh:.1f}",
                model_hit=clf_hit,
                model_direction=clf_direction,
                hit_threshold=hit_thresh,
                direction_threshold=dir_thresh,
            )
            comparison.evaluate_strategy(strategy, X_val, y_val)
    
    # ================== РЕЗУЛЬТАТЫ ==================

    min_trades_part = 0.01  # Минимальная доля трейдов от общего количества для отображения стратегии
    
    # Вывод топ-10 стратегий
    comparison.print_summary(top_n=10, min_trades_part=min_trades_part)  # Фильтр по минимальной частоте трейдов (например, 1%)
    
    # Детальное сравнение по precision
    print("\nDetailed comparison:")
    df = comparison.get_comparison_table(sort_by="precision", min_trades_part=min_trades_part)  # Фильтр по минимальной частоте трейдов (например, 1%)
    print(df.to_string(index=False))
    
    # Лучшая стратегия
    best = comparison.get_best_strategy("precision", min_trades_part=min_trades_part)  # Фильтр по минимальной частоте трейдов (например, 1%)
    print(f"\n{'='*80}")
    print(f"BEST STRATEGY: {best['strategy_name']}")
    print(f"  Type: {best['strategy_type']}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Trade Freq: {best['trade_freq']:.4f}")
    print(f"  N Trades: {best['n_trades']}")
    print(f"  Hyperparameters: {best['hyperparameters']}")
    print(f"{'='*80}\n")


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

    run_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        config=config
    )