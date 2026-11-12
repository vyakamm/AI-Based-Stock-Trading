"""Deep Learning Based Stock Trading Strategy.

A research framework for backtesting deep learning and machine learning
based trading strategies on historical OHLCV data.

Public API:
    data:          data loading utilities (local CSV / Yahoo Finance)
    models:        model wrappers (LSTM, RandomForest, XGBoost, SVM, KNN)
    strategies:    trading strategies and the strategy registry
    backtesting:   event-driven backtester with risk management + performance analytics
    optimization:  Optuna-based hyperparameter optimization
    pipeline:      high-level helpers to train / load models and build strategies
"""

from dl_trading import (
    backtesting,
    data,
    models,
    optimization,
    pipeline,
    strategies,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "backtesting",
    "data",
    "models",
    "optimization",
    "pipeline",
    "strategies",
]
