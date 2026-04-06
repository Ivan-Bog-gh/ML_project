# backtest/engine.py

"""
Основной движок бэктеста
- Event-driven симуляция
- No lookahead bias
- Triple barrier logic
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path

import sys
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from evaluation.strategies import TradingStrategy
# from backtest.portfolio import Portfolio


class BacktestEngine:
    """
    Простой детерминированный бэктест-движок
    
    Правила:
    - 1 позиция одновременно (no overlapping)
    - Triple barrier выход (TP/SL/timeout)
    - No leverage
    - Учет комиссий
    """
    
    def __init__(
        self,
        strategy: TradingStrategy,
        commission_rate: float = 0.001,
        initial_capital: float = 10000.0,
        mdd: float = 0.2,
    ):
        self.strategy = strategy
        self.commission_rate = commission_rate
        self.initial_capital = initial_capital
        self.mdd = mdd
    
    def run(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
            """
            Исполняет бэктест на основе готовых сигналов.
            df: исходные OHLCV
            signals_df: результат strategy.generate_signals() [signal, tp_price, sl_price, timeout_price, weight]
            """
            # Объединяем данные для удобства прохода в одном цикле
            data = pd.concat([df, signals_df], axis=1).dropna(subset=['signal'])
            data['next_open'] = data['open'].shift(-1).fillna(data['open'].iloc[-1])  # для расчета результата при таймауте

            trades = []
            active_position = None
            closed_position = False
            
            for timestamp, row in data.iterrows():
                # Если только закрыли позицию -> смотрим на equity и выходим в случае пересечения mdd
                if closed_position:
                    if len(trades) == 1:  # если это была первая сделка, то equity = initial_capital * (1 + pnl)
                        equity = self.initial_capital * (1 + trades[-1]['pnl'])
                    else:
                        equity = trades[-2]['equity'] * (1 + trades[-1]['pnl'])  # equity после закрытия последней сделки
                    trades[-1]['equity'] = equity  # сохраняем equity после закрытия сделки
                        
                    if equity <= self.initial_capital * (1 - self.mdd):
                        print(f"Stop out at {timestamp} with equity {equity:.2f}")
                        break
                    closed_position = False

                # 1. Если есть активная позиция — проверяем условия выхода
                if active_position:
                    # Проверка TP/SL (используем High/Low для точности)
                    if active_position['direction'] == 1:
                        if row['high'] >= active_position['tp_price']:
                            trades.append(self._close_trade(timestamp, active_position['tp_price'], active_position, "TP"))
                            active_position = None
                            closed_position = True
                        elif row['low'] <= active_position['sl_price']:
                            trades.append(self._close_trade(timestamp, active_position['sl_price'], active_position, "SL"))
                            active_position = None
                            closed_position = True
                    else: # Short
                        if row['low'] <= active_position['tp_price']:
                            trades.append(self._close_trade(timestamp, active_position['tp_price'], active_position, "TP"))
                            active_position = None
                            closed_position = True
                        elif row['high'] >= active_position['sl_price']:
                            trades.append(self._close_trade(timestamp, active_position['sl_price'], active_position, "SL"))
                            active_position = None
                            closed_position = True
                    
                    # Проверка таймаута (если позиция еще жива)
                    if active_position and timestamp >= active_position['timeout_time']:
                        trades.append(self._close_trade(timestamp, active_position['timeout_price'], active_position, "Timeout"))
                        active_position = None
                        closed_position = True

                # 2. Если позиции нет и есть новый сигнал — входим
                if not active_position and row['signal'] != 0:
                    # коррекция для SL. Цель в случае ЛЮБОГО убытка: -1% от депозита
                    sl_prob_result = (2 * self.commission_rate + abs(row['sl_price'] - row['next_open']) / row['next_open'])
                    weight_sl = 0.01 / sl_prob_result 
                    active_position = {
                        'entry_time': timestamp,
                        'entry_price': row['next_open'], # Вход на открытии следующей свечи
                        'direction': row['signal'],
                        'tp_price': row['tp_price'],
                        'sl_price': row['sl_price'],
                        'timeout_price': row['timeout_price'],
                        'timeout_time': row['timeout_time'],
                        'weight': row.get('weight', 1.0),
                        'weight_sl': weight_sl,
                        'ev': row['ev'],
                    }
            trades_df = pd.DataFrame(trades)
            self.trades_df = trades_df
            return self._compute_results()

    def _close_trade(self, exit_time, exit_price, position, reason):
        # Расчет доходности с учетом направления и веса (объем = Капитал * вес) + дальности SL (целевой SL: -1% от депозита)
        raw_return = (exit_price / position['entry_price'] - 1) * position['direction']
        net_return = (raw_return - 2 * self.commission_rate) * position['weight'] * position['weight_sl']  # учитываем комиссию, вес и коррекцию для SL
        
        return {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'exit_reason': reason,
            'pnl': net_return,
            'leverage': position['weight'] * position['weight_sl'],  # эффективный объем позиции от объема капитала
            **position,
        }
    
    def _compute_results(self) -> Dict:
        """Считает итоговые метрики"""
        trades_df = self.trades_df
        equity = trades_df["equity"]
        
        if len(trades_df) == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "ev": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }
        
        # Базовые метрики
        n_trades = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum()
        win_rate = wins / n_trades
        avg_pnl_adj = (trades_df["pnl"] / trades_df['leverage']).mean()
        avg_ev = trades_df["ev"].mean()
        
        # Return метрики
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (на equity curve)
        if len(equity) > 1:
            returns = equity.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 288) if returns.std() > 0 else 0  # 5min bars
        else:
            sharpe_ratio = 0.0
        
        # Max Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Exit reason distribution
        exit_reasons = trades_df["exit_reason"].value_counts(normalize=True).to_dict()
        
        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_pnl_adj": avg_pnl_adj,
            "avg_ev": avg_ev, 
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "exit_reasons": exit_reasons,
            "trades_df": trades_df,
            "equity_df": equity,
        }