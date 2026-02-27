# evaluation/strategies.py

"""
Wrapper для торговых стратегий.
Каждая стратегия инкапсулирует:
- Модель(и)
- Decision rules (thresholds, EV logic, etc.)
- Метод generate_signals() → итоговые сигналы
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any


class TradingStrategy(ABC):
    """
    Базовый класс для торговой стратегии.
    Стратегия должна уметь генерировать итоговые сигналы {-1, 0, +1}
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Обучение стратегии"""
        pass
    
    @abstractmethod
    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Генерирует итоговые торговые сигналы
        
        Returns:
            Series с сигналами {-1, 0, +1}
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Возвращает гиперпараметры стратегии"""
        pass


class BaselineStrategy(TradingStrategy):
    """
    Стратегия с одной мультиклассовой моделью
    """
    
    def __init__(
        self,
        name: str,
        model,  # BaselineClassifier
        threshold: float = 0.5,
    ):
        super().__init__(name)
        self.model = model
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self
    
    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Генерирует сигналы на основе threshold
        """
        return self.model.generate_signals(X, threshold=self.threshold)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "model_type": type(self.model).__name__,
        }


class TwoStageStrategy(TradingStrategy):
    """
    Стратегия с двумя моделями (hit + direction)
    """
    
    def __init__(
        self,
        name: str,
        model_hit,  # BaselineClassifier для hit/no-hit
        model_direction,  # BaselineClassifier для long/short
        hit_threshold: float = 0.5,
        direction_threshold: float = 0.5,
    ):
        super().__init__(name)
        self.model_hit = model_hit
        self.model_direction = model_direction
        self.hit_threshold = hit_threshold
        self.direction_threshold = direction_threshold
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучает обе модели
        """
        # Hit model: {0, 1}
        y_hit = y.replace(-1, 1)
        self.model_hit.fit(X, y_hit)
        
        # Direction model: {-1, +1} только на hit-samples
        mask_hit = y != 0
        X_hit = X[mask_hit]
        y_direction = y[mask_hit]
        self.model_direction.fit(X_hit, y_direction)
        
        return self
    
    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Двухэтапная генерация:
        1. Предсказываем hit/no-hit
        2. Для hit → предсказываем long/short
        """
        # Шаг 1: hit or not?
        probs_hit = self.model_hit.predict_proba(X)
        p_hit = probs_hit[:, 1]  # вероятность hit
        
        # Шаг 2: direction (только для hit)
        probs_direction = self.model_direction.predict_proba(X)
        
        # Получаем индексы классов
        if self.model_direction.calibrate:
            classes_dir = self.model_direction.pipeline.calibrated_classifiers_[0].estimator.named_steps['model'].classes_
        else:
            classes_dir = self.model_direction.pipeline.named_steps['model'].classes_
        
        idx_long = np.where(classes_dir == 1)[0][0]
        idx_short = np.where(classes_dir == -1)[0][0]
        
        p_long_given_hit = probs_direction[:, idx_long]
        p_short_given_hit = probs_direction[:, idx_short]
        
        # Итоговые сигналы
        signals = pd.Series(0, index=X.index)
        
        # Hit filter
        hit_mask = p_hit > self.hit_threshold
        
        # Direction filter (только для hit)
        long_mask = hit_mask & (p_long_given_hit > self.direction_threshold) & (p_long_given_hit > p_short_given_hit)
        short_mask = hit_mask & (p_short_given_hit > self.direction_threshold) & (p_short_given_hit > p_long_given_hit)
        
        signals[long_mask] = 1
        signals[short_mask] = -1
        
        return signals
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "hit_threshold": self.hit_threshold,
            "direction_threshold": self.direction_threshold,
            "hit_model_type": type(self.model_hit).__name__,
            "direction_model_type": type(self.model_direction).__name__,
        }


class EVStrategy(TradingStrategy):
    """
    Стратегия на основе Expected Value
    """
    
    def __init__(
        self,
        name: str,
        ev_trader,  # ExpectedValueTrader
        min_ev: float = 0.0,
        min_probability: float = 0.0,
        min_direction_confidence: float = 0.0,
        min_hit_confidence: float = 0.0,
    ):
        super().__init__(name)
        self.ev_trader = ev_trader
        self.min_ev = min_ev
        self.min_probability = min_probability
        self.min_direction_confidence = min_direction_confidence
        self.min_hit_confidence = min_hit_confidence
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        EV trader содержит уже обученные модели
        Здесь ничего не делаем
        """
        return self
    
    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Генерирует сигналы через EV logic
        """
        return self.ev_trader.generate_signals(
            X,
            min_ev=self.min_ev,
            min_probability=self.min_probability,
            min_direction_confidence=self.min_direction_confidence,
            min_hit_confidence=self.min_hit_confidence,
        )
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "min_ev": self.min_ev,
            "min_probability": self.min_probability,
            "min_direction_confidence": self.min_direction_confidence,
            "min_hit_confidence": self.min_hit_confidence,
            "tp_reward": self.ev_trader.tp_reward,
            "sl_loss": self.ev_trader.sl_loss,
            "commission": self.ev_trader.commission,
        }