# ML_project
End-to-end ML-based crypto trading system (5-min bars) with strict leakage control, walk-forward validation, transaction cost modeling, and risk-aware decision logic.

## Problem Statement

- 	This project aims to predict the probability of tradeable price movements, not future prices.
	- 	Formally: Given information available at time t, the model estimates whether the market will move by at least a fixed percentage threshold in either direction within a predefined future time horizon.
- 	This is intentionally not a point-forecasting problem, as point estimates are unstable and weakly actionable in financial markets.


## Project Organization
------------
│
│
├── ingestion/
│   ├── fetch_ohlcv.py
│
├── data/
│   ├── raw/               	# raw OHLCV
│   ├── interim/         	# cleaned, aligned
│   ├── processed/         	# ready to feaches/models
│
├── dataset/
│   ├── /__init__.py
│   ├── build_dataset.py   # assembling dataset
│   ├── splits.py          # split functions
│
├── preprocessing/
│   ├── /__init__.py
│   ├── clean_ohlcv.py          # data cleaning + base sanity
│   ├── align_time.py           # time alignment
│   ├── validate_ohlcv.py       # extra sanity checks
│   └── pipeline.py             # orchestrator
│
├── features/
│   ├── base_features.py
│   ├── build_features.py
│   ├── feature_cleaning.py
│
├── validation/
│   ├── check_features.py
│   ├── check_labels.py    # check label distribution
│
├── labeling/
│   ├── targets.py         # future return / event
│
├── models/
│   ├── baseline.py        # logistic / xgb
│   ├── pipeline.py        # sklearn Pipeline
│
├── evaluation/ 
│   ├── /__init__.py 
│   ├── metrics.py             # Base metrics (precision, ROC AUC, etc.) 
│   ├── calibration.py
│   ├── threshold_selection.py
│   └── comparison.py          # models comparison
│   └── strategies.py          # Description of strategy classes
│
├── backtest/
│   ├── engine.py
│   ├── costs.py
│   ├── metrics.py
│
├── risk/
│   ├── position_sizing.py
│   ├── kill_switch.py
│
├── api/
│   └── app.py             # FastAPI (позже)
│
├── config/
│   └── config.yaml
│
├── tests/
│   ├── test_features.py
│   └── test_backtest.py
│
└── README.md

------------


## Data Handling

Raw → Interim → Processed
The data pipeline follows a strict separation of responsibilities:

-	Raw: Original OHLCV data as received from the exchange API.

-	Interim: Cleaned data with:
	-	verified timestamp continuity
	-	repaired minor gaps
	-	integrity checks applied
	
-	Processed: Modeling-ready datasets:
	-	labels
	-	features (computed later)
	-	metadata required for training

This structure ensures reproducibility and traceability across all stages.


## Data Ingestion

Market data (OHLCV) is loaded incrementally from Binance using a dedicated ingestion script.

- 	Timeframe: 5-minute bars
- 	Storage: parquet files in `data/raw/`
- 	Updates: incremental (only new candles are fetched)

The ingestion process includes:
- 	schema validation
- 	strict time ordering
- 	duplicate removal
- 	explicit detection of data gaps

No forward-filling or interpolation is applied to avoid introducing look-ahead bias.


## Data Validation

Before feature generation, raw data is validated to ensure correctness and consistency.

The validation checks include:
- 	required schema and data types
- 	duplicate timestamps
- 	timestamp continuity
- 	OHLC consistency (high/low bounds, neg volume)
- 	missing or invalid values

Detected data gaps are logged but not automatically corrected.
This design choice preserves the integrity of the historical record and avoids artificial signals.


### Design Choices

The system intentionally avoids heavy orchestration (e.g., Airflow or streaming pipelines).
For research and prototyping, data ingestion is performed via a lightweight scheduled Python process.

In a production environment, this component can be replaced with a workflow orchestrator
without changing downstream feature engineering, modeling, or backtesting logic.

### Gap handling

Minor data gaps are forward-filled at the interim stage to preserve time continuity.
A dedicated flag (suspend_flag) is propagated to mark periods affected by exchange outages or abnormal trading conditions.
No labels are altered due to gaps; instead, downstream feature vectors are selectively excluded during training.

### suspend_flag semantics

suspend_flag = 1 indicates that one or more past observations used for feature computation originate from a suspended or artificially reconstructed market state.

-	Labels are preserved.
-	Feature vectors depending on these periods are excluded from training and evaluation.
-	This prevents training on distorted market regimes while retaining historical ground truth.


## Label Definition

### Event-based labeling

Labels are constructed using an event-based approach.
For each timestamp t, the future price path within a fixed horizon of N bars is examined:

-	the maximum price reached
-	the minimum price reached

Returns are computed relative to the close price at t.

### TP / SL logic

A fixed symmetric threshold is applied:

- 	+1 (long event)
	If the future maximum price exceeds the current price by at least the threshold before reaching -SL.
-	-1 (short event)
	If the future minimum price falls below the current price by at least the threshold before reaching +TP.
-	0 (no-trade)
	If neither threshold is reached within the horizon.
- 	+2 (sample dropped, minority)
	If the future maximum and minimum prices exceed the current price by at least the threshold at the same time.

This formulation mirrors a simplified TP/SL trading setup and ensures that labels correspond to realistically executable market events.

### Meaning of class 0

Class 0 represents non-tradeable market states.
It doesn't mean:

-	price did not move
-	market was flat

It means that no directional movement exceeded transaction-cost-adjusted thresholds within the given horizon, making any trade statistically unjustified under the defined setup.


### Modeling Philosophy

#### No oversampling

No oversampling, undersampling, or synthetic data generation techniques (e.g. SMOTE) are applied.
Such methods distort the temporal and probabilistic structure of financial time series and invalidate backtesting results.

#### Class imbalance is intentional

Class imbalance is a natural property of financial markets:

-	most market states are not tradeable	(label = 0)
-	profitable events are inherently rare

The model preserves this imbalance and handles it via:

-	cost-sensitive learning
-	probability thresholding
-	evaluation metrics focused on precision and stability

#### Focus on precision and stability, not accuracy

Model performance is not evaluated using raw accuracy.
Instead, emphasis is placed on:

-	precision of trade signals
-	robustness across time periods
-	stability of expected returns

The primary goal is risk-adjusted decision quality, not classification symmetry.

#### Closing note

This project is designed to reflect real-world quantitative modeling constraints rather than benchmark-driven machine learning setups. 
All design decisions prioritize causal validity, execution realism, and long-term robustness over short-term metric optimization.


## Feature Engineering v1

### Scope and intent

-	The first version of feature engineering focuses on causal, low-dimensional, and interpretable features derived strictly from historical market data.
-	The goal of this stage is not to maximize predictive performance, but to construct a clean and reliable feature space suitable for time-series modeling under real trading constraints.

### Feature groups

The following feature groups are included in v1:

-	Returns:
	-	Log-returns over multiple backward-looking windows
	-	Capture short- and medium-term momentum

-	Volatility:
	-	ATR-based volatility (normalized)
	-	Rolling standard deviation of log-returns
	-	Used to characterize risk regimes

-	Range / Structure:
	-	Rolling high–low ranges
	-	Candle body-to-range ratios
	-	Distinguish compression vs expansion states

-	Liquidity / Location:
	-	Relative distance to recent rolling extrema
	-	Position within rolling price ranges
	-	Proxy for liquidity concentration and market positioning

-	Volume:
	-	Rolling z-scores of volume and dollar volume
	-	Confirmation of price movements

-	Time features:
	-	Cyclical encoding of hour-of-day and day-of-week
	-	Capture intraday and weekly market regimes

All features are computed using strictly past information only.

### Causality and leakage control

The feature pipeline enforces strict temporal causality:

-	No forward-looking rolling windows
-	No usage of future OHLCV values
-	Explicit warmup periods for rolling computations
-	Validation checks to detect accidental future dependency

Synthetic or reconstructed data periods (e.g. exchange outages) are tracked via a dedicated 'suspend_flag'.
Feature vectors whose lookback windows intersect such periods are excluded from training and evaluation.

### No model-based feature selection

No model-driven feature selection is applied at this stage.
Specifically:

-	No feature importance pruning
-	No recursive elimination
-	No target-driven filtering

This is intentional to avoid implicit overfitting and information leakage during early experimentation.

### Only statistical hygiene applied

Feature cleaning at v1 is limited to purely statistical sanity checks, including:

-	Removal of near-constant features
-	Removal of almost perfectly correlated duplicates
-	Detection (but not automatic removal) of unusually high feature–label correlations

These steps aim to eliminate degenerate or redundant inputs without altering the underlying information content.

### Design philosophy

Feature Engineering v1 is designed to be:

-	conservative
-	reproducible
-	interpretable
-	execution-aware

More complex transformations (nonlinear interactions, regime-dependent features, meta-features) are intentionally deferred to later stages after establishing a robust baseline.


## Feature Engineering v1.1 — Extensions

- The main purpose of expanding the list of features is to improve the quality of the model by covering other parameters of the market condition.

### New feature groups

Building on v1, the following feature groups were added in v1.1.

-	Volatility regime positioning — rolling percentile rank of the 20-bar log-return standard deviation over longer windows [60, 120 bars]. 
	Captures where current volatility sits within its historical distribution, enabling regime-aware signal filtering.

-	Volatility expansion — ratio and difference between short-horizon (10-bar) and medium-horizon (20-bar) realized volatility. 
	Identifies vol compression and expansion states without introducing future dependency.

-	Volatility of volatilit — rolling standard deviation of the 20-bar vol series over a 20-bar window. 
	Measures instability of the volatility regime itself; elevated values indicate structurally uncertain market conditions.

-	Realized skewness — rolling skewness of log-returns over [20, 60] bar windows. 
	Asymmetric return distributions are associated with directional pressure and are not captured by symmetric volatility measures.

-	Intrabar close position — rolling mean of (close − low) / (high − low) over the same windows. 
	A price-action proxy for within-bar directional conviction, computed entirely from available OHLC data.

All features are computed using strictly past information. Warmup periods and `suspend_flag` propagation follow the same logic as v1.

Config changes: `percentile_windows: [60, 120]` added under `volatility`; new `realized_skewness` block with `windows: [20, 60]`.


## Dataset Assembly

A dedicated `dataset/` module handles the assembly of modeling-ready datasets.

- `build_dataset.py` — joins feature matrix with labels and metadata, enforces index alignment, drops rows invalidated by `suspend_flag`
- `splits.py` — provides walk-forward train/validation split functions consistent with the no-leakage requirement

This layer sits between feature engineering and model training, keeping each stage independently testable and reproducible.


## Models

`models/baseline.py` implements `BaselineClassifier`: a sklearn `Pipeline` (StandardScaler → estimator) defaulting to logistic regression. 
Optional probability calibration is available via `CalibratedClassifierCV` with time-series-safe cross-validation.

`ExpectedValueTrader` is a decision layer on top of two fitted classifiers:

	- `model_hit` — binary classifier: will price move at all? {0, 1}
	- `model_direction` — directional classifier conditioned on a hit: {−1, +1}

Expected value is computed as:
```
	EV_long  = P(long)  × R − P(short) × L − commission
	EV_short = P(short) × R − P(long)  × L − commission
```

Signals are generated only when EV exceeds a configurable minimum threshold.


## Evaluation Framework

The `evaluation/` module provides a structured approach to strategy comparison.

-	Metrics (`metrics.py`) — overall precision, long/short precision split, trade frequency, number of trades, F1, ROC AUC.

-	Calibration (`calibration.py`) — reliability diagrams and Expected Calibration Error (ECE).
	Well-calibrated probabilities are a prerequisite for valid EV computation.

-	Threshold selection (`threshold_selection.py`) — systematic sweep over probability thresholds to identify the precision / trade-frequency frontier.

-	Strategies (`strategies.py`) — abstract `TradingStrategy` base class with three concrete implementations:

	- `BaselineStrategy` — single multiclass model with a probability threshold
	- `TwoStageStrategy` — sequential hit filter followed by direction prediction
	- `EVStrategy` — signal generation via expected value with configurable minimum EV and minimum probability constraints

-	Comparison (`comparison.py`) — `StrategyComparison` class that evaluates a list of strategies against a validation set, stores results, and produces ranked summary tables. 
	Supports grid search over strategy hyperparameters via `compare_strategies_grid`.
	
### Design note

Strategy hyperparameters (thresholds, minimum EV) are selected on the
validation fold only. No strategy parameter influences the training procedure,
preserving the integrity of walk-forward evaluation.


## Model Research Log

-	This section documents modeling experiments conducted after the v1.1 baseline.
-	All results are compared against the reference baseline trained on fixed symmetric labels with a single 3-class logistic regression.

**Baseline reference:** `precision: 0.199`, `trade_freq: 0.0167`

Results are recorded regardless of outcome. Negative results are considered informative and retained for reproducibility.


### Experiment 1 — Volatility-adaptive label thresholds

**Motivation.**
	Fixed TP/SL thresholds treat all market states equally.
	The hypothesis was that scaling the barrier by current volatility (ATR-based) would produce classes that are more statistically homogeneous across regimes, 	making the classification problem easier.

**Implementation.**
	The fixed `threshold` in `labeling/targets.py` was replaced with a per-bar adaptive threshold proportional to a rolling ATR estimate.
	Labels were regenerated and the model was retrained on identical features.

**Result.**
	ROC AUC declined. Predicted probability distributions degraded.
	`precision: 0.161`, `trade_freq: 0.0066`.

**Conclusion.**
	Classes became more uniform statistically, but less predictable economically.
	The most likely cause: vol-scaling introduced collinearity between label construction and existing volatility features. 
	The barrier is now a function of ATR, which is already present in the feature matrix — this reduces the	marginal information content of volatility features and weakens model calibration. 
	Vol-scaled labels require either volatility-orthogonal features or a fundamentally different feature set.


### Experiment 2 — Two-stage model: hit vs. no-hit + direction

**Motivation.**
	A single 3-class classifier must simultaneously learn two distinct decisions:
	-	whether a tradeable move will occur, and in which direction. 
	Decomposing this into two simpler binary tasks was expected to improve calibration and allow independent tuning of each decision boundary.

**Implementation.**
	`BaselineClassifier` was retained as the base estimator for both models.
	- Model 1 (`model_hit`): binary, trained on `{0, 1}` where `±1 → 1`.
	- Model 2 (`model_direction`): binary, trained on `{−1, +1}` restricted to hit samples only.
	An `ExpectedValueTrader` layer combined both outputs into final signals.
	Several supplementary market-state features were added (see v1.1 extensions).

**Result.**
	Model flexibility increased. 
	Precision improved marginally over Experiment 1 but did not surpass the baseline. 
	`precision: 0.174`, `trade_freq: 0.011`.

**Conclusion.**
	Training on all market states indiscriminately limits the signal available to both models. 
	In low-information market conditions (majority of bars), the direction model has no meaningful basis for a decision. 
	The two-stage architecture is sound, but requires a preceding market-state filter to restrict the sample to periods where alpha is plausibly present.


### Experiment 3 — CUSUM pre-filter for market state selection

**Motivation.**
	A core assumption of all prior experiments is that every bar is a valid training candidate. 
	In practice, most 5-minute bars occur during low-activity periods where no directional signal exists. 
	A CUSUM-based event filter (following Lopez de Prado, *Advances in Financial Machine Learning*) was applied to isolate bars where cumulative price deviation exceeded a threshold, restricting training and inference to periods of detectable market activity.

**Implementation.**
	A symmetric CUSUM filter was implemented over log-returns with a tuned threshold. 
	Triggered events reduced the active sample from 100% of bars to approximately 3%. 
	Models from Experiment 2 were retrained on this filtered dataset.

**Result.**
	ROC AUC on the direction model collapsed. Precision and trade frequency shifted significantly: 
	`precision: 0.1384`, `trade_freq: 0.1334`.

**Conclusion.**
	At ~3% of the original sample size, the feature matrix is too high-dimensional for the available observations. 
	The model cannot generalize: the ratio of features to training samples is unfavorable. 
	Two corrective directions are identified:

	- **Loosen the filter** — use a lower CUSUM threshold to retain a larger fraction of bars while still excluding the quietest periods.
	- **Reduce feature dimensionality** — apply explicit feature selection before filtering to keep only the most informative inputs given the reduced sample.

The CUSUM pre-filtering concept is retained as a structural component of the pipeline. 
Tuning of filter sensitivity is deferred to the next iteration.


### Summary

| Experiment | Key change | Precision | Trade Freq | vs. Baseline |
|---|---|---|---|---|
| Baseline | Fixed labels, 3-class logistic | 0.199 | 0.0167 | — |
| Exp 1 | Vol-adaptive labels | 0.161 | 0.0066 | ↓ |
| Exp 2 | Two-stage hit + direction | 0.174 | 0.0110 | ↓ |
| Exp 3 | CUSUM filter + two-stage | 0.138 | 0.1334 | ↓ |

**Primary bottleneck identified:** 
	the model is trained and evaluated on all market states without qualitative discrimination. 
	Meaningful alpha extraction likely requires either a well-calibrated regime filter or a fundamentally different label construction that is orthogonal to the existing feature set.