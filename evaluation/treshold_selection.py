# evaluation/threshold_selection.py

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.metrics import precision_score


def find_threshold_for_precision(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_precision: float,
    class_index: int = 1,
    search_range: Tuple[float, float] = (0.5, 0.95),
    n_steps: int = 50,
) -> Tuple[float, Dict[str, float]]:
    """
    Находит threshold для достижения целевого precision
    
    Args:
        y_true: истинные метки
        y_proba: вероятности (N x n_classes)
        target_precision: желаемый precision (например, 0.6)
        class_index: индекс класса для которого ищем threshold
        search_range: диапазон поиска threshold
        n_steps: количество шагов в поиске
    
    Returns:
        (best_threshold, metrics)
    """
    thresholds = np.linspace(search_range[0], search_range[1], n_steps)
    
    best_threshold = None
    best_diff = float('inf')
    best_metrics = None
    
    for thresh in thresholds:
        # Генерируем предсказания
        preds = (y_proba[:, class_index] > thresh).astype(int)
        
        if preds.sum() == 0:
            continue
        
        # Считаем precision только для этого класса
        mask = preds == 1
        if mask.sum() == 0:
            continue
            
        precision = (y_true[mask] == class_index).mean()
        trade_freq = mask.mean()
        
        # Ищем ближайший к целевому
        diff = abs(precision - target_precision)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh
            best_metrics = {
                "precision": precision,
                "trade_freq": trade_freq,
                "n_trades": mask.sum(),
                "threshold": thresh,
            }
    
    return best_threshold, best_metrics


def find_threshold_for_trade_frequency(
    y_proba: np.ndarray,
    target_trade_freq: float,
    class_index: int = 1,
) -> Tuple[float, float]:
    """
    Находит threshold для достижения желаемой частоты трейдов
    
    Args:
        y_proba: вероятности (N x n_classes)
        target_trade_freq: желаемая частота (например, 0.3 = 30% трейдов)
        class_index: индекс класса
    
    Returns:
        (threshold, actual_trade_freq)
    """
    probs = y_proba[:, class_index]
    
    # Находим квантиль, соответствующий желаемой частоте
    # Если хотим 30% трейдов, берем 70-й перцентиль
    threshold = np.quantile(probs, 1 - target_trade_freq)
    
    # Проверяем фактическую частоту
    actual_freq = (probs > threshold).mean()
    
    return threshold, actual_freq


def precision_recall_curve_trading(
    y_true: pd.Series,
    y_proba: np.ndarray,
    classes: np.ndarray,
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """
    Строит precision-recall-trade_freq кривую для торговых сигналов
    
    Args:
        y_true: истинные метки
        y_proba: вероятности
        classes: массив классов модели
        thresholds: список threshold для проверки
    
    Returns:
        DataFrame с метриками для каждого threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 50)
    
    results = []
    
    idx_long = np.where(classes == 1)[0][0]
    idx_short = np.where(classes == -1)[0][0]
    
    for thresh in thresholds:
        # Генерируем сигналы
        signals = pd.Series(0, index=y_true.index)
        signals[y_proba[:, idx_long] > thresh] = 1
        signals[y_proba[:, idx_short] > thresh] = -1
        
        traded = signals != 0
        
        if traded.sum() == 0:
            continue
        
        precision = precision_score(
            y_true[traded],
            signals[traded],
            average="macro",
            zero_division=0,
        )
        
        trade_freq = traded.mean()
        
        results.append({
            "threshold": thresh,
            "precision": precision,
            "trade_freq": trade_freq,
            "n_trades": traded.sum(),
        })
    
    return pd.DataFrame(results)