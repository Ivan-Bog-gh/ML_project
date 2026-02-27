# evaluation/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)
from typing import Dict


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Комплексная оценка классификатора
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        y_proba: вероятности (опционально, для ROC AUC и Brier)
    
    Returns:
        dict с метриками
    """
    metrics = {}
    
    # Базовые метрики
    metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    # Вероятностные метрики
    if y_proba is not None:
        # ROC AUC
        if len(np.unique(y_true)) == 2:  # binary
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            # Brier score (calibration)
            metrics["brier_score"] = brier_score_loss(y_true, y_proba[:, 1])
        else:  # multiclass
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, 
                multi_class='ovr', 
                average='macro'
            )
    
    return metrics


def evaluate_trading_signals(
    y_true: pd.Series,
    signals: pd.Series,
) -> Dict[str, float]:
    """
    Оценка торговых сигналов (с классом 0 = no trade)
    
    Args:
        y_true: истинные метки (-1, 0, 1)
        signals: торговые сигналы (-1, 0, 1)
    
    Returns:
        dict с метриками
    """
    traded = signals != 0
    n_trades = traded.sum()
    trade_freq = traded.mean()
    
    if n_trades == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "trade_freq": 0.0,
            "n_trades": 0,
        }
    
    # Метрики только на торгованных сделках
    precision = precision_score(
        y_true[traded],
        signals[traded],
        average="macro",
        zero_division=0,
    )
    
    recall = recall_score(
        y_true[traded],
        signals[traded],
        average="macro",
        zero_division=0,
    )
    
    f1 = f1_score(
        y_true[traded],
        signals[traded],
        average="macro",
        zero_division=0,
    )
    
    # Точность по направлениям
    long_mask = signals == 1
    short_mask = signals == -1
    
    long_precision = (y_true[long_mask] == 1).mean() if long_mask.sum() > 0 else 0
    short_precision = (y_true[short_mask] == -1).mean() if short_mask.sum() > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trade_freq": trade_freq,
        "n_trades": n_trades,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "n_long": long_mask.sum(),
        "n_short": short_mask.sum(),
    }