# models/baseline.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone


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
        self.base_estimator = estimator # для последующего воспроизведения аналогичной модели
        
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
        self.calibration_method = calibration_method
        self.cv = cv

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
        y_cnt = len(y.unique())
        if y_cnt > 2:
            roc_auc = roc_auc_score(y, probs, multi_class='ovo') # из-за дисбаланса классов
        elif y_cnt == 2:
            roc_auc = roc_auc_score(y, probs[:, 1])
        else:            
            roc_auc = 0.0  # Если только один класс, AUC не определен
        
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
    
    def copy(self):
        init_params = {
            "estimator": clone(self.base_estimator),
            "calibrate": self.calibrate,
            "calibration_method": self.calibration_method,
            "cv": self.cv if self.calibrate else None,
        }
        return BaselineClassifier(**init_params)

class BarrierEstimator:
    """
    Вычисляет динамические TP/SL барьеры на основе волатильности.
    
    Вход:  DataFrame с OHLCV + (опционально) предпосчитанная волатильность
    Выход: DataFrame с колонками [tp_perc, sl_perc, net_reward, net_loss]
    """
    def __init__(
        self,
        com_rate: float = 0.001,      # ставка комиссии (0.1%)
        atr_window: int = 12,         # окно для ATR (по сути брать из config)
        k: float = 1.8,               # множитель для ATR (по сути брать из config)
        tp_atr_mult: float = 1.0,     # TP = tp_atr_mult * ATR
        sl_atr_mult: float = 1.0      # SL = sl_atr_mult * ATR
    ):
        self.com_rate = com_rate
        self.atr_window = atr_window
        self.k = k
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает DataFrame с барьерами по каждому бару.
        Внутри считает ATR.
        
        Колонки результата:
            close        — цена закрытия (для расчета sl/tp-price в будущем)
            tp_perc      — TP барьер (% от цены)
            sl_perc      — SL барьер (% от цены)
            next_open    — цена открытия следующей свечи (для расчета ТВХ и результата при таймауте)
            timeout_result      — фактический результат при достижении таймаута (открытие следующей свечой)
            timeout_time        — timeout барьер
            net_reward   — R после комиссии
            net_loss     — L после комиссии
            vol          — волатильность (для диагностики)
            commission          — стоимость комиссии
            commission_ratio    — доля комиссии от TP (диагностика)
        """
    
        # Расчет log returns и volatility
        log_returns = np.log(df["close"] / df["close"].shift(1))
        vol = log_returns.rolling(window=self.atr_window).std() * 1.8 # !!! тянуть также из конфига ?
        timeout_result = df["close"].shift(-self.atr_window) / df["open"].shift(-1) - 1 # Как по факту закрылись
        timeout_time = df.index.to_series().shift(-self.atr_window).fillna(df.index[-1])
        # timeout_time = min(df.index.shift(-self.atr_window), df.index[-1]) # Когда закрылись (для диагностики)
        
        tp_perc = self.k * vol * self.tp_atr_mult * np.sqrt(self.atr_window)
        sl_perc = self.k * vol * self.sl_atr_mult * np.sqrt(self.atr_window)
        
        # Комиссия уменьшает reward и увеличивает loss (!!! не учтен вход на открытии следующей свечи)
        net_reward = (tp_perc - 2 * self.com_rate)
        net_loss   = (sl_perc + 2 * self.com_rate)
        
        return pd.DataFrame({
            "close":            df["close"],
            "tp_perc":          tp_perc,
            "sl_perc":          sl_perc,
            "next_open":        df["open"].shift(-1),
            "timeout_result":   timeout_result, # от направления зависит, складывать или вычитать с комиссией
            "timeout_time":     timeout_time,
            "net_reward":       net_reward,
            "net_loss":         net_loss,
            # Удобство для отладки
            "vol":              vol,
            "commission":       (2 * self.com_rate),
            "commission_ratio": (2 * self.com_rate) / tp_perc,
        }, index=df.index)