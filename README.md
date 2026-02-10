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