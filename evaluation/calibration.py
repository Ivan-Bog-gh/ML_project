# evaluation/calibration.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from typing import Dict, Any


def evaluate_calibration(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Оценивает калибровку вероятностей
    
    Args:
        y_true: истинные метки (бинарные)
        y_proba: предсказанные вероятности
        n_bins: количество бинов
        strategy: стратегия биннинга ("uniform" или "quantile")
    
    Returns:
        dict с метриками калибровки
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba,
        n_bins=n_bins,
        strategy=strategy
    )
    
    # Expected Calibration Error (ECE)
    ece = np.abs(prob_true - prob_pred).mean()
    
    # Maximum Calibration Error (MCE)
    mce = np.abs(prob_true - prob_pred).max()
    
    return {
        "ece": ece,
        "mce": mce,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def plot_calibration_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
):
    """
    Строит calibration curve
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba,
        n_bins=n_bins,
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(prob_pred, prob_true, 'o-', label=model_name)
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title(f'Calibration Curve: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()