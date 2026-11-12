"""Registry of available strategies.

The registry is the single source of truth used by both the CLI and the
optimizer. Each entry maps a lowercase strategy name to:

- ``type``: the :class:`~dl_trading.strategies.base.StrategyType`,
- ``class``: importable dotted path of the strategy class,
- ``suggest_params``: an Optuna hyperparameter sampler returning a dict that
    always contains ``past_history``.

Strategy names double as artifact suffixes: trained models and hyperparameter
files are saved as
``model_{SYMBOL}_{STRATEGY}.h5/.pkl`` and
``best_hyperparams_{SYMBOL}_{STRATEGY}.xml``.
"""

from __future__ import annotations

import importlib
from typing import Any

from dl_trading.strategies.base import StrategyType


def load_class_from_string(class_path: str) -> type:
    """Import a class from its dotted path.

    Args:
        class_path: Dotted path, e.g. ``"dl_trading.strategies.base.RSIStrategy"``.

    Returns:
        The imported class object.

    Raises:
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Optuna parameter samplers
# ---------------------------------------------------------------------------
def _suggest_ma_params(trial: Any) -> dict[str, Any]:
    """Sample moving-average windows with the constraint short < long."""
    short_window = trial.suggest_int("short_window", 5, 20)
    long_window = trial.suggest_int("long_window", short_window + 1, 100)
    return {"short_window": short_window, "long_window": long_window, "past_history": long_window}


def _suggest_rsi_params(trial: Any) -> dict[str, Any]:
    """Sample RSI thresholds and window."""
    window = trial.suggest_int("window", 10, 30)
    return {
        "window": window,
        "overbought": trial.suggest_int("overbought", 70, 90),
        "oversold": trial.suggest_int("oversold", 10, 30),
        # One extra bar is needed by the first diff().
        "past_history": window + 1,
    }


def _suggest_randomforest_params(trial: Any) -> dict[str, Any]:
    """Sample RandomForest hyperparameters."""
    return {
        "past_history": trial.suggest_int("past_history", 10, 100),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }


def _suggest_xgboost_params(trial: Any) -> dict[str, Any]:
    """Sample XGBoost hyperparameters."""
    return {
        "past_history": trial.suggest_int("past_history", 10, 100),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    }


def _suggest_svm_params(trial: Any) -> dict[str, Any]:
    """Sample SVM hyperparameters."""
    return {
        "past_history": trial.suggest_int("past_history", 10, 100),
        "C": trial.suggest_float("C", 0.001, 1000.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
    }


def _suggest_knn_params(trial: Any) -> dict[str, Any]:
    """Sample KNN hyperparameters."""
    return {
        "past_history": trial.suggest_int("past_history", 10, 100),
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }


def _suggest_lstm_params(trial: Any) -> dict[str, Any]:
    """Sample LSTM architecture and training hyperparameters."""
    return {
        "past_history": trial.suggest_int("past_history", 10, 100),
        "stack_depth": trial.suggest_int("stack_depth", 1, 5),
        "number_units_per_layer": trial.suggest_int("number_units_per_layer", 16, 256),
        "epochs": trial.suggest_int("epochs", 10, 100),
        "batch_size": trial.suggest_int("batch_size", 32, 128),
    }


STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "ma": {
        "type": StrategyType.TECHNICAL_INDICATOR,
        "class": "dl_trading.strategies.base.MovingAverageCrossoverStrategy",
        "suggest_params": _suggest_ma_params,
    },
    "rsi": {
        "type": StrategyType.TECHNICAL_INDICATOR,
        "class": "dl_trading.strategies.base.RSIStrategy",
        "suggest_params": _suggest_rsi_params,
    },
    "randomforest": {
        "type": StrategyType.MACHINE_LEARNING,
        "class": "dl_trading.strategies.base.RandomForestStrategy",
        "suggest_params": _suggest_randomforest_params,
    },
    "xgboost": {
        "type": StrategyType.MACHINE_LEARNING,
        "class": "dl_trading.strategies.base.XGBoostStrategy",
        "suggest_params": _suggest_xgboost_params,
    },
    "svm": {
        "type": StrategyType.MACHINE_LEARNING,
        "class": "dl_trading.strategies.base.SVMStrategy",
        "suggest_params": _suggest_svm_params,
    },
    "knn": {
        "type": StrategyType.MACHINE_LEARNING,
        "class": "dl_trading.strategies.base.KNNStrategy",
        "suggest_params": _suggest_knn_params,
    },
    "lstm": {
        "type": StrategyType.DEEP_LEARNING,
        "class": "dl_trading.strategies.base.LSTMStrategy",
        "suggest_params": _suggest_lstm_params,
    },
}


def list_strategies() -> list[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(STRATEGY_REGISTRY)
