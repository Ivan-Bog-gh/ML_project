# models/baseline.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


class BaselineClassifier:
    """
    Model layer сборки двух моделей (hit + direction).
    Считает ML-метрики и генерирует сигналы на основе вероятностей.
     - hit vs no hit (binary classification)
     - direction (long vs short) при условии hit
    """
    def __init__(
        self,
        estimator=None,
        class_weight="balanced",
        calibrate=False,
        calibration_method="sigmoid",
        cv=None,
    ):
        if estimator is None:
            estimator = LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                solver="lbfgs",
                random_state=42,
            )
        
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        
        # Калибровка вероятностей (опционально)
        if calibrate:
            if cv is None:
                cv = TimeSeriesSplit(n_splits=3)
            self.pipeline = CalibratedClassifierCV(
                self.pipeline, 
                method=calibration_method,  # "isotonic" или "sigmoid"
                cv=cv
            )
        
        self.calibrate = calibrate

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict (self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def generate_signals(self, X: pd.DataFrame, threshold: float) -> pd.Series:
        probs = self.predict_proba(X)
        
        if self.calibrate:
            # Для калиброванной модели classes_ находится глубже
            classes = self.pipeline.calibrated_classifiers_[0].estimator.named_steps['model'].classes_
        else:
            classes = self.pipeline.named_steps['model'].classes_

        # class order: [-1, 0, +1]
        p_long = probs[:, classes == 1].ravel()
        p_short = probs[:, classes == -1].ravel()

        signals = pd.Series(0, index=X.index)
        signals[p_long > threshold] = 1
        signals[p_short > threshold] = -1

        return signals
    
    def evaluate_roc_auc(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """ROC AUC для бинарной классификации"""
        probs = self.predict_proba(X)
        roc_auc = roc_auc_score(y, probs[:, 1])
        
        return {"roc_auc": roc_auc}

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
        signals = self.generate_signals(X, threshold)

        traded = signals != 0
        if traded.sum() == 0:
            return {
                "precision": 0.0,
                "trade_freq": 0.0,
            }

        precision = precision_score(
            y[traded],
            signals[traded],
            average="macro",
            zero_division=0,
        )

        trade_freq = traded.mean()

        return {
            "precision": precision,
            "trade_freq": trade_freq,
        }

class ExpectedValueTrader:
    """
    Decision layer поверх двух моделей (hit + direction).
    Считает Expected Value и принимает решения на основе EV.
    """
    def __init__(
        self,
        model_hit: BaselineClassifier,
        model_direction: BaselineClassifier,
        tp_reward: float = 1.0,  # R (reward при достижении TP)
        sl_loss: float = 1.0,    # L (loss при достижении SL)
        commission: float = 0.001,  # комиссия за сделку
    ):
        self.model_hit = model_hit
        self.model_direction = model_direction
        self.tp_reward = tp_reward
        self.sl_loss = sl_loss
        self.commission = commission
    
    def compute_probabilities(self, X: pd.DataFrame) -> dict:
        """
        Вычисляет условные вероятности:
        - P(hit) - вероятность того, что цена достигнет границы
        - P(long|hit) - вероятность long при условии hit
        - P(long) = P(hit) * P(long|hit)
        - P(short) = P(hit) * (1 - P(long|hit))
        """
        p_hit = self.model_hit.predict_proba(X)[:, 1]  # вероятность hit (класс 1 из [0, 1])
        p_long_given_hit = self.model_direction.predict_proba(X)[:, 1]  # P(long|hit) (класс 1 из [-1, 1])
        
        p_long = p_hit * p_long_given_hit
        p_short = p_hit * (1 - p_long_given_hit)
        p_no_trade = 1 - p_hit
        
        return {
            "p_hit": p_hit,
            "p_long_given_hit": p_long_given_hit,
            "p_long": p_long,
            "p_short": p_short,
            "p_no_trade": p_no_trade,
        }
    
    def compute_expected_value(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        EV_long = P(long) * R - P(short) * L - commission
        EV_short = P(short) * R - P(long) * L - commission
        
        Где:
        - R = reward (TP)
        - L = loss (SL)
        """
        probs = self.compute_probabilities(X)
        
        ev_long = (
            probs["p_long"] * self.tp_reward 
            - probs["p_short"] * self.sl_loss 
            - self.commission
        )
        
        ev_short = (
            probs["p_short"] * self.tp_reward 
            - probs["p_long"] * self.sl_loss 
            - self.commission
        )
        
        return pd.DataFrame({
            "ev_long": ev_long,
            "ev_short": ev_short,
            "p_long": probs["p_long"],
            "p_short": probs["p_short"],
            "p_hit": probs["p_hit"],
        }, index=X.index)
    
    def generate_signals(self, X: pd.DataFrame, min_ev: float = 0.0, min_probability: float = 0.0) -> pd.Series:
        """
        Генерирует сигналы на основе Expected Value:
        - Long если EV_long > min_ev и EV_long > EV_short
        - Short если EV_short > min_ev и EV_short > EV_long
        - 0 иначе
        """
        ev_df = self.compute_expected_value(X)
        
        signals = pd.Series(0, index=X.index)
        
        # Long: EV_long максимален и положителен
        long_mask = (ev_df["ev_long"] > min_ev) & (ev_df["ev_long"] > ev_df["ev_short"])
        if min_probability > 0:
            long_mask &= (ev_df["p_long"] > min_probability * ev_df["p_hit"])  # Условие на минимальную вероятность long при условии hit
        signals[long_mask] = 1
        
        # Short: EV_short максимален и положителен
        short_mask = (ev_df["ev_short"] > min_ev) & (ev_df["ev_short"] > ev_df["ev_long"])
        if min_probability > 0:
            short_mask &= (ev_df["p_short"] > min_probability * ev_df["p_hit"])  # Условие на минимальную вероятность short при условии hit
        signals[short_mask] = -1
        
        return signals
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, min_ev: float = 0.0, min_probability: float = 0.0) -> dict:
        """Оценка производительности EV-based стратегии"""
        signals = self.generate_signals(X, min_ev, min_probability)
        
        traded = signals != 0
        if traded.sum() == 0:
            return {
                "precision": 0.0,
                "trade_freq": 0.0,
                "mean_ev": 0.0,
            }
        
        precision = precision_score(
            y[traded],
            signals[traded],
            average="macro",
            zero_division=0,
        )
        
        trade_freq = traded.mean()
        
        ev_df = self.compute_expected_value(X)
        mean_ev_long = ev_df.loc[signals == 1, "ev_long"].mean() if (signals == 1).sum() > 0 else 0
        mean_ev_short = ev_df.loc[signals == -1, "ev_short"].mean() if (signals == -1).sum() > 0 else 0
        
        return {
            "precision": precision,
            "trade_freq": trade_freq,
            "mean_ev_long": mean_ev_long,
            "mean_ev_short": mean_ev_short,
        }