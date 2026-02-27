# evaluation/comparison.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import json

from evaluation.metrics import evaluate_trading_signals
from evaluation.strategies import TradingStrategy, BaselineStrategy, TwoStageStrategy, EVStrategy


class StrategyComparison:
    """
    Класс для сравнения торговых СТРАТЕГИЙ
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_strategy(
        self,
        strategy: TradingStrategy,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Оценивает ИТОГОВЫЕ сигналы стратегии
        
        Args:
            strategy: торговая стратегия
            X_val: валидационные features
            y_val: валидационные labels
            metadata: дополнительная информация
        
        Returns:
            dict с результатами
        """
        # Генерируем итоговые сигналы
        signals = strategy.generate_signals(X_val)
        
        # Оцениваем сигналы
        metrics = evaluate_trading_signals(y_val, signals)
        
        # Формируем результат
        result = {
            "strategy_name": strategy.name,
            "strategy_type": type(strategy).__name__,
            **metrics,
            "hyperparameters": strategy.get_hyperparameters(),
        }
        
        if metadata:
            result["metadata"] = metadata
        
        self.results.append(result)
        
        return result
    
    def evaluate_strategies(
        self,
        strategies: List[TradingStrategy],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """
        Оценивает список стратегий
        """
        for strategy in strategies:
            print(f"\nEvaluating: {strategy.name}")
            result = self.evaluate_strategy(strategy, X_val, y_val)
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Trade Freq: {result['trade_freq']:.4f}")
            print(f"  N Trades: {result['n_trades']}")
    
    def get_comparison_table(
        self,
        sort_by: str = "precision",
        columns: List[str] = None,
        min_trades_part: float = None,
    ) -> pd.DataFrame:
        """
        Возвращает таблицу сравнения стратегий
        """
        df = pd.DataFrame(self.results)
        
        if columns is None:
            columns = [
                "strategy_name",
                "strategy_type",
                "precision",
                "trade_freq",
                "n_trades",
                "long_precision",
                "short_precision",
                "f1",
            ]
        
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
            
        if min_trades_part is not None:
            df = df[df["trade_freq"] >= min_trades_part]
        
        return df
    
    def get_best_strategy(self, metric: str = "precision", min_trades_part: float = None) -> Dict[str, Any]:
        """
        Возвращает лучшую стратегию по заданной метрике
        """        
        df = self.get_comparison_table(sort_by=metric, min_trades_part=min_trades_part)
        if df.empty:
            return None
        return self.results[df.index[0]]
    
    def print_summary(self, top_n: int = 10, min_trades_part: float = None):
        """
        Выводит топ-N стратегий
        """
        df = self.get_comparison_table(sort_by="precision", min_trades_part=min_trades_part)
        
        if df.empty:
            print("No strategies to display.")
            return
        
        print("\n" + "="*100)
        print(f"STRATEGY COMPARISON - TOP {min(top_n, len(df))} BY PRECISION")
        print("="*100)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head(top_n).to_string(index=False))
        print("="*100 + "\n")
    
    def save_results(self, filepath: Path):
        """Сохраняет результаты"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def load_results(self, filepath: Path):
        """Загружает результаты"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)


def compare_strategies_grid(
    base_strategy: TradingStrategy,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Dict[str, List[Any]],
    strategy_class: type,
) -> StrategyComparison:
    """
    Grid search по гиперпараметрам стратегии
    
    Args:
        base_strategy: базовая стратегия (с обученными моделями)
        X_val: валидационные данные
        y_val: валидационные labels
        param_grid: сетка параметров
        strategy_class: класс стратегии
    
    Returns:
        StrategyComparison с результатами
    """
    comparison = StrategyComparison()
    
    # Генерируем все комбинации параметров
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        
        # Создаем стратегию с этими параметрами
        if strategy_class == BaselineStrategy:
            strategy = BaselineStrategy(
                name=f"Baseline_thresh{params['threshold']:.2f}",
                model=base_strategy.model,
                **params
            )
        elif strategy_class == TwoStageStrategy:
            strategy = TwoStageStrategy(
                name=f"TwoStage_hit{params['hit_threshold']:.2f}_dir{params['direction_threshold']:.2f}",
                model_hit=base_strategy.model_hit,
                model_direction=base_strategy.model_direction,
                **params
            )
        elif strategy_class == EVStrategy:
            strategy = EVStrategy(
                name=f"EV_minp{params.get('min_probability', 0):.2f}",
                ev_trader=base_strategy.ev_trader,
                **params
            )
        else:
            raise ValueError(f"Unknown strategy class: {strategy_class}")
        
        comparison.evaluate_strategy(strategy, X_val, y_val)
    
    return comparison