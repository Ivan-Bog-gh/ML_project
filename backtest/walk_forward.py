# backtest/walk_forward.py

"""
Walk-Forward Validation
- Train на первых 60% → test на 60-80%
- Retrain на 0-80% → test на 80-100%
"""


import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import ParameterGrid
from pathlib import Path

import sys
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_performance_metrics
from evaluation.comparison import StrategyComparison
from evaluation.strategies import TradingStrategy


class WalkForwardAnalysis:
    """
    Walk-forward validation для выбора оптимальной стратегии на каждом периоде
    """
    
    def __init__(
        self,
        # strategy,  # Выбранная стратегия (BaselineStrategy, EVStrategy, etc.)
        models=None,  # Словарь с моделями для hit и direction (опционально, если стратегия сама не обучает модели)
        com_rate=0.001,
        initial_capital=10000.0,
        mdd=0.2,
    ):
        # self.strategy = strategy
        self.models = models
        self.com_rate = com_rate
        self.initial_capital = initial_capital
        self.mdd = mdd
        self.results = []
    
    def run(
        self,
        strategies: Dict[str, TradingStrategy],  # Словарь стратегий
        X: pd.DataFrame,
        y: pd.Series,
        df_ohlc: pd.DataFrame,
        barriers: pd.DataFrame,
        splits: List[Tuple[float, float, float, float, float]],
    ) -> pd.DataFrame:
        """
        Запускает walk-forward выборку стратегий по созданным моделям
        
        Args:
            strategies: словарь стратегий для сравнения (например, {"Baseline": BaselineStrategy(), "EV": EVStrategy()})
            X: features
            y: labels
            df_ohlc: OHLC данные
            barriers: данные с барьерами для расчета EV
            splits: список (train_start, train_end, test_end)
            strategy_kwargs: параметры для стратегии
        
        Returns:
            DataFrame с результатами каждого fold
        """
        n = len(X)
        possible_strategies = set() # Для определения всех ТС, которые допускались в прошлом (чтобы в новом периоде не использовать недопустимую ранее ТС)
        
        for fold_idx, (train_part, val_part, test_part, start_range, end_range) in enumerate(splits):
            # Определяем границы от всей выборки для текущего fold
            train_start = start_range
            train_end = start_range + (end_range - start_range) * train_part
            val_end = start_range + (end_range - start_range) * (train_part + val_part)
            test_end = end_range

            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx + 1}: Train [{train_start:.0%}-{train_end:.0%}] → Val [{train_end:.0%}-{val_end:.0%}] → Test [{val_end:.0%}-{test_end:.0%}]")
            print(f"{'='*80}")
            
            # Индексы split
            train_start_idx = int(n * train_start)
            train_end_idx = int(n * train_end)
            val_end_idx = int(n * val_end)
            test_end_idx = int(n * test_end)
            
            X_train = X.iloc[train_start_idx:train_end_idx]
            y_train = y.iloc[train_start_idx:train_end_idx]
            barriers_train = barriers.iloc[train_start_idx:train_end_idx]
            X_val = X.iloc[train_end_idx:val_end_idx]
            y_val = y.iloc[train_end_idx:val_end_idx]
            df_ohlc_val = df_ohlc.iloc[train_end_idx:val_end_idx]
            barriers_val = barriers.iloc[train_end_idx:val_end_idx]
            X_test = X.iloc[val_end_idx:test_end_idx]
            y_test = y.iloc[val_end_idx:test_end_idx]
            df_ohlc_test = df_ohlc.iloc[val_end_idx:test_end_idx]
            barriers_test = barriers.iloc[val_end_idx:test_end_idx]

            print(f"Train size: {len(X_train):,} |  Val size: {len(X_val):,} |  Test size: {len(X_test):,}")

            # Инициализируем аналогичные модели, чтобы не перезаписывать оригинал (важно для WFA) и обучаем их на train-данных
            curr_models = {}
            model_hit = self.models.get("model_hit", {})
            if model_hit:
                model_hit = model_hit.copy()  # создаем копию модели для текущего fold
                y_train_hit = y_train.replace(-1, 1)  # 0 = no-hit, 1 = hit (long or short)
                model_hit.fit(X_train, y_train_hit)
                curr_models["model_hit"] = model_hit

            model_direction = self.models.get("model_direction", {})
            if model_direction:                
                model_direction = model_direction.copy()  # создаем копию модели для текущего fold
                
                # create dataset for directional classifier (только hit)
                bin_mask_train = y_train != 0
                X_train_bin, y_train_bin = X_train[bin_mask_train], y_train[bin_mask_train]
                model_direction.fit(X_train_bin, y_train_bin)
                curr_models["model_direction"] = model_direction


            model_baseline = self.models.get("model_baseline", {})
            if model_baseline:
                model_baseline = model_baseline.copy()  # создаем копию модели для текущего fold
                model_baseline.fit(X_train, y_train)
                curr_models["model_baseline"] = model_baseline

            # Выбираем стратегию для текущего fold
            comparison = StrategyComparison()
            for strategy_info in strategies.values():
                print(f"\nEvaluating strategy: {strategy_info['name']}")
                required_models = strategy_info.get("required_models", [])
                if not all(model in curr_models for model in required_models):
                    print(f"Skipping strategy {strategy_info['name']} due to missing models: {set(required_models) - set(curr_models.keys())}")
                    continue # если для стратегии не хватает обученных моделей, пропускаем ее

                hyperparams_list = ParameterGrid(strategy_info.get("hyperparams", {}))
                for strategy_hyperparams in hyperparams_list:
                    hp_txt = "_" + "_".join([f"{k}={v}" for k, v in strategy_hyperparams.items()])
                    strategy = strategy_info["model"](
                        name=strategy_info["name"] + hp_txt,
                        **{model: curr_models[model] for model in required_models},  # передаем обученные модели в стратегию
                        **strategy_hyperparams
                    )
                    strategy.fit(X_train, y_train, barriers_train) # Обучение стратегии
                    comparison.evaluate_strategy(strategy, X_val, y_val, barriers_val)

            # Выбор лучшеий стратегии на валидации (по EV)
            min_trades_part = 0.008  # Минимальная доля трейдов от общего количества для отображения стратегии
    
            comparison.print_summary(top_n=10, min_trades_part=min_trades_part)  # Фильтр по минимальной частоте трейдов (например, 1%)
            
            # Лучшая стратегия с учетом всех допущенных в этом fold стратегий (по EV) для тестирования на unseen данных
            all_possible_strategies = set(comparison.get_comparison_table(min_trades_part=min_trades_part)["strategy_name"].unique()) # Все стратегии, которые допускались в этом цикле (по результатам валидации)
            if fold_idx == 0:
                possible_strategies = all_possible_strategies # На первом цикле инициализируем полный список возможных стратегий
            else:
                possible_strategies = possible_strategies.intersection(all_possible_strategies) # На следующих циклах сужаем список возможных стратегий пересечением с новыми допущенными стратегиями
            best = comparison.get_best_strategy("EV", min_trades_part=min_trades_part, possible_strategies=possible_strategies)  # Фильтр по минимальной частоте трейдов (например, 1%)
            best_strategy = best['strategy_class']( # создаем обновленный экземпляр лучшей стратегии для теста на unseen данных
                name=best['strategy_name'],
                **best['hyperparameters']
            )

            print(f"\n{'='*80}")
            print(f"BEST STRATEGY: {best['strategy_name']}")
            print(f"  Type: {best['strategy_type']}")
            print(f"  Precision: {best['precision']:.4f}")
            print(f"  Trade Freq: {best['trade_freq']:.4f}")
            print(f"  N Trades: {best['n_trades']}")
            print(f"  Hyperparameters: {best['hyperparameters']}")
            print(f"{'='*80}\n")

            #    
            best_strategy.fit(X_train, y_train, barriers_train)
            signals_test = best_strategy.generate_signals(X=X_test, barriers=barriers_test)

            # Не хватает показать: стабильность И не-overfitting обученных моделей / соответствие факта (avg_pnl) и прогноза (avg_EV) на OOS
            models_params_train = best_strategy.evaluate_models(X=X_train, y=y_train, prefix="train_")
            models_params_test = best_strategy.evaluate_models(X=X_test, y=y_test, prefix="test_")

            # Бэктест на test
            engine = BacktestEngine(
                strategy=best_strategy,
                commission_rate=self.com_rate,
                initial_capital=self.initial_capital,
                mdd=self.mdd,
            )
            
            backtest_results = engine.run(df=df_ohlc_test, signals_df=signals_test)
            
            # Собираем результаты
            fold_result = {
                "fold": fold_idx + 1,
                "train_period": f"{train_start:.0%}-{train_end:.0%}",
                "val_period": f"{train_end:.0%}-{val_end:.0%}",
                "test_period": f"{val_end:.0%}-{test_end:.0%}",
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                **backtest_results,
                **models_params_train,
                **models_params_test,
            }
            
            self.results.append(fold_result)
            
            # Выводим результаты fold
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Trades: {backtest_results['n_trades']}")
            print(f"  Win Rate: {backtest_results['win_rate']:.2%}")
            print(f"  Total Return: {backtest_results['total_return']:.2%}")
            print(f"  Sharpe: {backtest_results['sharpe_ratio']:.2f}")
            print(f"  Max DD: {backtest_results['max_drawdown']:.2%}")
        
        return self._create_summary()
    
    def _create_summary(self) -> pd.DataFrame:
        """
        Создает сводную таблицу результатов
        """
        summary_data = []
        
        for result in self.results:

            models_params = dict() # Для оценки модели
            for key in result.keys():
                if key.endswith('_roc_auc'):
                    models_params[key] = result[key]

            summary_data.append({
                "fold": result["fold"],
                "test_period": result["test_period"],

                "n_trades": result["n_trades"],
                "win_rate": result["win_rate"],
                "total_return": result["total_return"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown": result["max_drawdown"],
                
                "avg_pnl_adj": result["avg_pnl_adj"],
                "avg_ev": result["avg_ev"],

                **models_params,
            })
        
        df = pd.DataFrame(summary_data)
        
        # Добавляем статистику по fold'ам
        print(f"\n{'='*80}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        print(f"\n📊 Stability Metrics:")
        print(f"  Mean Win Rate:      {df['win_rate'].mean():.2%} ± {df['win_rate'].std():.2%}")
        print(f"  Mean Return:        {df['total_return'].mean():.2%} ± {df['total_return'].std():.2%}")
        print(f"  Mean Sharpe:        {df['sharpe_ratio'].mean():.2f} ± {df['sharpe_ratio'].std():.2f}")
        print(f"  Worst DD:           {df['max_drawdown'].min():.2%}")
        print(f"{'='*80}\n")
        
        return df