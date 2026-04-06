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

from models.baseline import BaselineClassifier


class TradingStrategy(ABC):
    """
    Базовый класс для торговой стратегии.
    Стратегия должна уметь генерировать итоговые сигналы {-1, 0, +1}
    """
    
    def __init__(self, name: str):
        self.name = name
        self.ev_trade_timeout_long_baseline = 0.0 # Будет рассчитан в fit()
        self.ev_trade_timeout_short_baseline = 0.0 # Будет рассчитан в fit()
        self.ev_all_trade_baseline = 0.0 # Будет рассчитан в fit()

    @abstractmethod
    def _get_clean_probs(self, X: pd.DataFrame) -> dict:
        """
        Вычисляет условные вероятности классов:
        - P(long)       = вероятность того, что цена достигнет верхней границы
        - P(short)      = вероятность того, что цена достигнет нижней границы
        - P(no_trade)   = вероятность того, что цена не достигнет границы
        """
        pass
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        barriers_train: pd.DataFrame, # выход BarrierEstimator.fit()
        quantile: float = 0.5,  # медиана по умолчанию; < 0.5 — консервативнее
    ) -> float:
        """
        Собирает EV на тренировочных срабатываниях и сохраняет внутри калибровочные параметры.

        Логика:
            1. Генерируем сигналы на train (без поправки ev_false_trade, чтобы не зациклиться)
            2. Находим: 
                2.1 Ложные срабатывания: сигнал был, но направление неверное (отдельно для long/short на случай разного drift)
                2.2 Все срабатывания: сигнал был, независимо от результата
            3. Считаем EV этих баров и берём агрегат (медиана / квантиль)
                3.1 EV всех срабатываний для оценки weight в generate_signals (сравнение с baseline)

        Параметры:
            quantile: квантиль агрегации EV ложных срабатываний.
                      0.5 = медиана (нейтральный выбор).
                      0.25 = консервативно (занижаем поправку).

        Возвращает:
            Вычисленный ev_trade_timeout_(long/short)_baseline (также сохраняется в self).
            Вычисленный ev_all_trade_baseline (также сохраняется в self) для оценки веса по сделке.
        """
        # Считаем EV без поправки (ev_false_trade=0), чтобы не было рекурсии
        signals_df = self.generate_signals(X_train, barriers_train)#, _ev_df=ev_df)  # генерируем сигналы на основе EV с ev_false_trade=0
        signals = signals_df["signal"]

        # Ложные срабатывания: сигнал был, но y != signal
        traded_mask = signals != 0
        if traded_mask.sum() == 0:
            self.ev_trade_timeout_long_baseline = 0.0
            self.ev_trade_timeout_short_baseline = 0.0
            self.ev_all_trade_baseline = 0.0
            return 0.0

        false_signals_mask = traded_mask & (y_train == 0) # сигнал был, но не было движения (hit = 0)

        if false_signals_mask.sum() == 0:
            # Ложных срабатываний нет — поправка не нужна
            self.ev_trade_timeout_long_baseline = 0.0
            self.ev_trade_timeout_short_baseline = 0.0
            self.ev_all_trade_baseline = 0.0
            return 0.0

        # фактический EV ложных срабатываний на train: берём EV_long
        long_false_mask = false_signals_mask & (signals == 1)
        ev_long_false = pd.Series(index=X_train.index[long_false_mask], dtype=float)
        ev_long_false.loc[X_train.index[long_false_mask]] = barriers_train.loc[X_train.index[long_false_mask], "timeout_result"] - barriers_train.loc[X_train.index[long_false_mask], "commission"]
        self.ev_trade_timeout_long_baseline = float(ev_long_false.quantile(quantile))

        # фактический EV ложных срабатываний на train: берём EV_short
        short_false_mask = false_signals_mask & (signals == -1)
        ev_short_false = pd.Series(index=X_train.index[short_false_mask], dtype=float)
        ev_short_false.loc[X_train.index[short_false_mask]] = - barriers_train.loc[X_train.index[short_false_mask], "timeout_result"] - barriers_train.loc[X_train.index[short_false_mask], "commission"]
        self.ev_trade_timeout_short_baseline = float(ev_short_false.quantile(quantile))

        # EV всех срабатываний для подсчета weight в generate_signals (сравнение с baseline)
        ev_signals = pd.Series(index=X_train.index[traded_mask], dtype=float)
        
        sl_mask = traded_mask & (y_train != 0) & (y_train != signals)  # сигнал был, но направление неверное → EV = -L
        tp_mask = traded_mask & (y_train == signals)  # сигнал был, и направление верное → EV = R
        
        ev_signals.loc[X_train.index[long_false_mask]] = barriers_train.loc[X_train.index[long_false_mask], "timeout_result"] - barriers_train.loc[X_train.index[long_false_mask], "commission"]
        ev_signals.loc[X_train.index[short_false_mask]] = - barriers_train.loc[X_train.index[short_false_mask], "timeout_result"] - barriers_train.loc[X_train.index[short_false_mask], "commission"]
        ev_signals.loc[X_train.index[sl_mask]] = - barriers_train.loc[X_train.index[sl_mask], "net_loss"]
        ev_signals.loc[X_train.index[tp_mask]] = barriers_train.loc[X_train.index[tp_mask], "net_reward"]
        self.ev_all_trade_baseline = float(ev_signals.quantile(0.95)) # для оценки веса по сделке (можно брать более оптимистичный квантиль, чтобы не занижать вес)

        return self
    
    def compute_expected_value(
            self, 
            X: pd.DataFrame,
            barriers: pd.DataFrame, # PnL для каждого барьера (tp, sl, timeout) с учетом комиссии
        ) -> pd.DataFrame:
        """
        EV_long = P(long) * R - P(short) * L + P(trade_timeout) * EV_false_long_trade
        EV_short = P(short) * R - P(long) * L + P(trade_timeout) * EV_false_short_trade
        
        Где:
        - R = reward (TP)
        - L = loss (SL)
        """
        assert barriers.index.equals(X.index), "Индексы X и barriers должны совпадать"
        assert {"net_reward", "net_loss"}.issubset(barriers.columns)
        
        probs = self._get_clean_probs(X)
        R = barriers["net_reward"]
        L = barriers["net_loss"]
        ev_trade_timeout_long = self.ev_trade_timeout_long_baseline
        ev_trade_timeout_short = self.ev_trade_timeout_short_baseline

        ev_long = probs["p_long"] * R - probs["p_short"] * L + probs["p_no_trade"] * ev_trade_timeout_long
        ev_short = probs["p_short"] * R - probs["p_long"] * L + probs["p_no_trade"] * ev_trade_timeout_short
        
        return pd.DataFrame({
            "ev_long": ev_long,
            "ev_short": ev_short,
            "p_long": probs["p_long"],
            "p_short": probs["p_short"],
            "p_hit": probs["p_hit"],
            "p_no_trade": probs["p_no_trade"],
            # Полезно для диагностики — EV без поправки no_trade
            "ev_long_raw": probs["p_long"] * R - probs["p_short"] * L,
            "ev_short_raw": probs["p_short"] * R - probs["p_long"] * L,
            "commission_ratio": barriers["commission_ratio"],
            "ev_trade_timeout_long": ev_trade_timeout_long,
            "ev_trade_timeout_short": ev_trade_timeout_short,
        }, index=X.index)
    
    @abstractmethod
    def fit_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучение моделей стратегии (если нужно). Вызывается ДО fit(), чтобы модели были готовы к генерации сигналов на этапе калибровки.
        Используется на этапе WFA для обучения на расширяющемся окне.
        """
        pass
    
    @abstractmethod
    def generate_signals(self, X: pd.DataFrame, barriers: pd.DataFrame = None) -> pd.DataFrame:
        """
        Генерирует итоговые торговые сигналы
        
        Returns DataFrame:
            signal: сигналы {-1, 0, +1}
            tp_price: цена TP
            sl_price: цена SL
            weight: веса для входа в сделку на основе EV / p_success относительно показателей на train
        """
        pass

    @abstractmethod
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict:
        """
        Оценивает модели стратегии на заданных данных (например, ROC AUC) и возвращает словарь метрик с префиксом для различения моделей внутри стратегии (если их несколько).
        
        Returns Dict:
            метрики оценки моделей, например:
            {f"{prefix}roc_auc": roc_auc}
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
        model_baseline: BaselineClassifier,  # BaselineClassifier
        threshold: float = 0.5,
    ):
        super().__init__(name)
        self.model = model_baseline
        self.threshold = threshold

    def _get_clean_probs(self, X: pd.DataFrame) -> dict:
        """Вспомогательный метод-адаптер"""
        # Вероятность класса: 1 (long), класса -1 (short) и класса 0 (no trade)
        p_hit = self.model.predict_proba(X)
        
        if self.model.calibrate:
            # Для калиброванной модели classes_ находится глубже
            classes = self.model.pipeline.calibrated_classifiers_[0].estimator.named_steps['model'].classes_
        else:
            classes = self.model.pipeline.named_steps['model'].classes_

        p_long = p_hit[:, classes == 1].ravel() if 1 in classes else np.zeros(len(X))
        p_short = p_hit[:, classes == -1].ravel() if -1 in classes else np.zeros(len(X))
        p_no_trade = p_hit[:, classes == 0].ravel() if 0 in classes else np.zeros(len(X))
        max_prob = np.max(p_hit, axis=1)
        
        return {
            'p_long': p_long,
            'p_short': p_short,
            'p_hit': 1 - p_no_trade,
            'p_no_trade': p_no_trade,
            'max_prob': max_prob,
        }
    
    def fit_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Обучаем модель стратегии
        """
         # Инициализируем аналогичную модель, чтобы не перезаписывать оригинал (важно для WFA)
        model_params = self.model.get_init_params()
        self.model = BaselineClassifier(**model_params)
        
        self.model.fit(X_train, y_train)
        return self
    
    def generate_signals(self, X: pd.DataFrame, barriers: pd.DataFrame = None) -> pd.DataFrame:
        """
        Генерирует сигналы на основе threshold
        (!) Нужно актуализировать output (weight)
        """
        probs = self._get_clean_probs(X)
        signals = pd.DataFrame(index=X.index)
        
        # Direction filter (только для hit)
        long_mask = (probs['p_long'] > self.threshold) & (probs['p_long'] >= probs['max_prob'])
        short_mask = (probs['p_short'] > self.threshold) & (probs['p_short'] >= probs['max_prob'])
        
        signals['signal'] = 0
        signals.loc[long_mask, 'signal'] = 1
        signals.loc[short_mask, 'signal'] = -1
        
        signals['tp_price'] = barriers["close"] * (1 + barriers["tp_perc"] * signals['signal'])
        signals['sl_price'] = barriers["close"] * (1 - barriers["sl_perc"] * signals['signal'])
        signals['timeout_price'] = barriers["next_open"] * (1 + barriers["timeout_result"])
        signals['timeout_time'] = barriers["timeout_time"]
        
        # Расчет веса (возможно отрицательное значение, поскольку принятие решения по вероятностям, а не EV)
        ev_df = self.compute_expected_value(X, barriers)
        weight = pd.Series(1.0, index=X.index) # дефолтные веса
        ev = pd.Series(0.0, index=X.index)
        if self.ev_all_trade_baseline != 0:
            ev = np.where(signals['signal'] == 1, ev_df["ev_long"], np.where(signals['signal'] == -1, ev_df["ev_short"], 0))
            ev_train = ev - np.where(signals['signal'] == 1, self.ev_trade_timeout_long_baseline, self.ev_trade_timeout_short_baseline) * ev_df["p_no_trade"]  # учитываем поправку на таймауты
            weight = np.where(ev_train < 0, 0, ev_train) / self.ev_all_trade_baseline # Зануление позиции если EV < 0
        signals['weight'] = weight
        signals.loc[signals['weight'] == 0, 'signal'] = 0 # адаптация сигналов для корректности расчетов
        signals['ev'] = ev # Для оценки стратегии в рамках WFA (соответствие avg_pnl)
        
        return signals
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict:
        model_roc_auc = self.model.evaluate_roc_auc(X, y)
        return {
            f"{prefix}model_roc_auc": model_roc_auc["roc_auc"],
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "threshold": self.threshold,
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
    
    def _get_clean_probs(self, X: pd.DataFrame) -> dict:
        """
        Вычисляет условные вероятности:
        - P(hit) - вероятность того, что цена достигнет границы
        - P(long|hit) - вероятность long при условии hit
        - P(long)       = P(hit) * P(long|hit)
        - P(short)      = P(hit) * (1 - P(long|hit))
        - P(no_trade)   = 1 - P(hit)
        """
        p_hit = self.model_hit.predict_proba(X)[:, 1]  # вероятность hit (класс 1 из [0, 1])
        p_long_given_hit = self.model_direction.predict_proba(X)[:, 1]  # P(long|hit) (класс 1 из [-1, 1])
                
        return {
            "p_hit": p_hit,
            "p_long_given_hit": p_long_given_hit,
            "p_short_given_hit": 1 - p_long_given_hit,
            "p_long": p_hit * p_long_given_hit,
            "p_short": p_hit * (1 - p_long_given_hit),
            "p_no_trade": 1 - p_hit,
        }
    
    def fit_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Обучаем обе модели стратегии
        """
        # Инициализируем аналогичные модели, чтобы не перезаписывать оригинал (важно для WFA)
        model_hit_params = self.model_hit.get_init_params()
        self.model_hit = BaselineClassifier(**model_hit_params)
        model_direction_params = self.model_direction.get_init_params()
        self.model_direction = BaselineClassifier(**model_direction_params)

        # Hit model: {0, 1}
        y_hit = y_train.replace(-1, 1)
        self.model_hit.fit(X_train, y_hit)
        
        # Direction model: {-1, +1} только на hit-samples
        mask_hit = y_train != 0
        X_hit = X_train[mask_hit]
        y_direction = y_train[mask_hit]
        self.model_direction.fit(X_hit, y_direction)

        return self
    
    def generate_signals(self, X: pd.DataFrame, barriers: pd.DataFrame = None) -> pd.DataFrame:
        """
        Двухэтапная генерация:
        1. Предсказываем hit/no-hit
        2. Для hit → предсказываем long/short
        (!) Нужно актуализировать output (weight)
        """
        probs = self._get_clean_probs(X)
        
        # Итоговые сигналы
        signals = pd.DataFrame(index=X.index)
        
        # Direction filter (только для hit)
        long_mask = (probs['p_hit'] > self.hit_threshold) & (probs['p_long_given_hit'] > self.direction_threshold) & (probs['p_long'] > probs['p_short'])
        short_mask = (probs['p_hit'] > self.hit_threshold) & (probs['p_short_given_hit'] > self.direction_threshold) & (probs['p_short'] > probs['p_long'])
        
        signals['signal'] = 0
        signals.loc[long_mask, 'signal'] = 1
        signals.loc[short_mask, 'signal'] = -1

        signals['tp_price'] = barriers["close"] * (1 + barriers["tp_perc"] * signals['signal'])
        signals['sl_price'] = barriers["close"] * (1 - barriers["sl_perc"] * signals['signal'])
        signals['timeout_price'] = barriers["next_open"] * (1 + barriers["timeout_result"])
        signals['timeout_time'] = barriers["timeout_time"]
        
        # Расчет веса (возможно отрицательное значение, поскольку принятие решения по вероятностям, а не EV)
        ev_df = self.compute_expected_value(X, barriers)
        weight = pd.Series(1.0, index=X.index) # дефолтные веса
        ev = pd.Series(0.0, index=X.index)
        if self.ev_all_trade_baseline != 0:
            ev = np.where(signals['signal'] == 1, ev_df["ev_long"], np.where(signals['signal'] == -1, ev_df["ev_short"], 0))
            ev_train = ev - np.where(signals['signal'] == 1, self.ev_trade_timeout_long_baseline, self.ev_trade_timeout_short_baseline) * ev_df["p_no_trade"]  # учитываем поправку на таймауты
            weight = np.where(ev_train < 0, 0, ev_train) / self.ev_all_trade_baseline # Зануление позиции если EV < 0

        signals['weight'] = weight
        signals.loc[signals['weight'] == 0, 'signal'] = 0 # адаптация сигналов для корректности расчетов
        signals['ev'] = ev # Для оценки стратегии в рамках WFA (соответствие avg_pnl)
        # signals['weight'] = probs['p_hit'] / (self.p_hit_baseline if self.p_hit_baseline > 0 else 1)  # Пример веса — вероятность hit
        
        return signals
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict:
        y_hit = y.replace(-1, 1)  # для оценки hit-модели
        hit_model_roc_auc = self.model_hit.evaluate_roc_auc(X, y_hit)

        direction_mask = y != 0
        X_direction, y_direction = X[direction_mask], y[direction_mask]
        direction_roc = self.model_direction.evaluate_roc_auc(X_direction, y_direction)  # для оценки direction-модели
        return {
            f"{prefix}hit_model_roc_auc": hit_model_roc_auc["roc_auc"],
            f"{prefix}direction_model_roc_auc": direction_roc["roc_auc"],
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "model_hit": self.model_hit,
            "model_direction": self.model_direction,
            "hit_threshold": self.hit_threshold,
            "direction_threshold": self.direction_threshold,
        }


class EVStrategy(TradingStrategy):
    """
    Стратегия на основе Expected Value
    """
    
    def __init__(
        self,
        name: str,
        model_hit,
        model_direction,
        min_ev: float = 0.0,
        min_probability: float = 0.0,
    ):
        super().__init__(name)
        self.model_hit = model_hit
        self.model_direction = model_direction
        self.min_ev = min_ev
        self.min_probability = min_probability

    def _get_clean_probs(self, X: pd.DataFrame) -> dict:
        """
        Вычисляет условные вероятности:
        - P(hit) - вероятность того, что цена достигнет границы
        - P(long|hit) - вероятность long при условии hit
        - P(long)       = P(hit) * P(long|hit)
        - P(short)      = P(hit) * (1 - P(long|hit))
        - P(no_trade)   = 1 - P(hit)
        """
        p_hit = self.model_hit.predict_proba(X)[:, 1]  # вероятность hit (класс 1 из [0, 1])
        p_long_given_hit = self.model_direction.predict_proba(X)[:, 1]  # P(long|hit) (класс 1 из [-1, 1])
                
        return {
            "p_hit": p_hit,
            "p_long_given_hit": p_long_given_hit,
            "p_long": p_hit * p_long_given_hit,
            "p_short": p_hit * (1 - p_long_given_hit),
            "p_no_trade": 1 - p_hit,
        }
    
    def fit_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Обучаем обе модели стратегии
        """
        # Инициализируем аналогичные модели, чтобы не перезаписывать оригинал (важно для WFA)
        model_hit_params = self.model_hit.get_init_params()
        self.model_hit = BaselineClassifier(**model_hit_params)
        model_direction_params = self.model_direction.get_init_params()
        self.model_direction = BaselineClassifier(**model_direction_params)

        # Hit model: {0, 1}
        y_hit = y_train.replace(-1, 1)
        self.model_hit.fit(X_train, y_hit)
        
        # Direction model: {-1, +1} только на hit-samples
        mask_hit = y_train != 0
        X_hit = X_train[mask_hit]
        y_direction = y_train[mask_hit]
        self.model_direction.fit(X_hit, y_direction)
        
        # При переобученнии моделей нужно заново считать EV на train для корректного baseline
        self.ev_trade_timeout_long_baseline = 0.0
        self.ev_trade_timeout_short_baseline = 0.0
        self.ev_all_trade_baseline = 0.0

        return self

    def generate_signals(self, X, barriers) -> pd.DataFrame:
        
        ev_df = self.compute_expected_value(X, barriers)
        min_ev = self.min_ev
        min_probability = self.min_probability
        signals = pd.DataFrame(index=X.index)
        
        signals['signal'] = 0 # default no trade

        # Long: EV_long максимален и положителен
        long_mask = (ev_df["ev_long"] > min_ev) & (ev_df["ev_long"] > ev_df["ev_short"])
        if min_probability > 0:
            long_mask &= (ev_df["p_long"] > min_probability)  # Условие на минимальную вероятность long (абсолютная)
        signals.loc[long_mask, 'signal'] = 1
        
        # Short: EV_short максимален и положителен
        short_mask = (ev_df["ev_short"] > min_ev) & (ev_df["ev_short"] > ev_df["ev_long"])
        if min_probability > 0:
            short_mask &= (ev_df["p_short"] > min_probability)  # Условие на минимальную вероятность short (абсолютная)
        signals.loc[short_mask, 'signal'] = -1

        # Расчет веса для сделки (сравнение с EV на train)
        weight = pd.Series(1.0, index=X.index)  # по умолчанию вес 1
        ev = pd.Series(0.0, index=X.index)
        if self.ev_all_trade_baseline != 0:
            ev = np.where(signals['signal'] == 1, ev_df["ev_long"], np.where(signals['signal'] == -1, ev_df["ev_short"], 0))
            ev_train = ev - np.where(signals['signal'] == 1, self.ev_trade_timeout_long_baseline, self.ev_trade_timeout_short_baseline) * ev_df["p_no_trade"]  # учитываем поправку на таймауты
            weight = ev_train / self.ev_all_trade_baseline

        # Расчет целевых уровней TP/SL на основе ожидаемых tp/sl_perc + Формируем итоговый DataFrame
        signals['tp_price'] = barriers["close"] * (1 + barriers["tp_perc"] * signals['signal'])  # для short будет минус, для long — плюс
        signals['sl_price'] = barriers["close"] * (1 - barriers["sl_perc"] * signals['signal'])  # для short будет плюс, для long — минус
        signals['timeout_price'] = barriers["next_open"] * (1 + barriers["timeout_result"])  # для short будет плюс, для long — минус
        signals['timeout_time'] = barriers["timeout_time"]
        signals['weight'] = weight # Пример веса
        signals['ev'] = ev # Для оценки стратегии в рамках WFA (соответствие avg_pnl)
        
        return signals
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict:
        y_hit = y.replace(-1, 1)  # для оценки hit-модели
        hit_model_roc_auc = self.model_hit.evaluate_roc_auc(X, y_hit)

        direction_mask = y != 0
        X_direction, y_direction = X[direction_mask], y[direction_mask]
        direction_roc = self.model_direction.evaluate_roc_auc(X_direction, y_direction)  # для оценки direction-модели
        return {
            f"{prefix}hit_model_roc_auc": hit_model_roc_auc["roc_auc"],
            f"{prefix}direction_model_roc_auc": direction_roc["roc_auc"],
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "model_hit": self.model_hit,
            "model_direction": self.model_direction,
            "min_ev": self.min_ev,
            "min_probability": self.min_probability,
        }
