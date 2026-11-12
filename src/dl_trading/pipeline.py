"""High-level helpers to train, load and persist models and strategies.

This module is the glue between the low-level building blocks (data,
models, strategies) and the CLI. It owns:

- the artifact naming convention (``model_{SYMBOL}_{STRATEGY}``,
  ``best_hyperparams_{SYMBOL}_{STRATEGY}``),
- the model factory driven by the strategy registry,
- the train / load lifecycle for ML and DL models,
- strategy construction from a hyperparameter dictionary.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from dl_trading.data.loader import load_hyperparameters
from dl_trading.data.preprocessing import prepare_data_for_dl, prepare_data_for_ml
from dl_trading.strategies.base import Strategy, StrategyType
from dl_trading.strategies.registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ArtifactPaths:
    """Locations of the artifacts for one (symbol, strategy) pair."""

    hyperparams: Path
    model: Path

    @property
    def model_exists(self) -> bool:
        return self.model.is_file()


def artifact_paths(
    symbol: str,
    strategy_name: str,
    artifacts_dir: str | Path = "artifacts",
) -> ArtifactPaths:
    """Compute the canonical artifact paths for a (symbol, strategy) pair.

    Args:
        symbol: Ticker symbol, e.g. ``"AMZN"``.
        strategy_name: Registered strategy name, e.g. ``"lstm"``.
        artifacts_dir: Root directory holding ``models/`` and
            ``hyperparams/`` sub-directories.

    Returns:
        An :class:`ArtifactPaths` instance.

    Raises:
        KeyError: If ``strategy_name`` is not registered.
    """
    strategy_name = strategy_name.lower()
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{strategy_name}'. Available: {sorted(STRATEGY_REGISTRY)}")

    root = Path(artifacts_dir)
    return ArtifactPaths(
        hyperparams=root / "hyperparams" / f"best_hyperparams_{symbol}_{strategy_name}.xml",
        model=root / "models" / f"model_{symbol}_{strategy_name}",
    )


def build_model(strategy_name: str, hyperparams: dict[str, Any]) -> Any:
    """Instantiate an (unfitted) model matching ``strategy_name``.

    Args:
        strategy_name: Registered strategy name.
        hyperparams: Hyperparameters (may contain extra keys such as
            ``stop_loss`` or ``past_history`` which are ignored here).

    Returns:
        A fresh model wrapper (``LSTMModel``, ``RandomForestModel``, ...).

    Raises:
        NotImplementedError: For technical-indicator strategies (no model).
    """
    if strategy_name == "lstm":
        from dl_trading.models.lstm_model import LSTMModel

        return LSTMModel(
            input_shape=(2, hyperparams["past_history"]),
            lstm_layer_units=hyperparams["number_units_per_layer"],
            stack_depth=hyperparams["stack_depth"],
        )
    if strategy_name == "randomforest":
        from dl_trading.models.sklearn_models import RandomForestModel

        return RandomForestModel(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth"),
        )
    if strategy_name == "xgboost":
        from dl_trading.models.sklearn_models import XGBoostModel

        return XGBoostModel(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.1),
        )
    if strategy_name == "svm":
        from dl_trading.models.sklearn_models import SVMModel

        return SVMModel(
            kernel=hyperparams.get("kernel", "rbf"),
            C=hyperparams.get("C", 1.0),
        )
    if strategy_name == "knn":
        from dl_trading.models.sklearn_models import KNNModel

        return KNNModel(
            n_neighbors=hyperparams.get("n_neighbors", 5),
            weights=hyperparams.get("weights", "uniform"),
        )
    raise NotImplementedError(f"No model to build for strategy '{strategy_name}'.")


def train_model(
    strategy_name: str,
    hyperparams: dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[Any, dict[str, Any]]:
    """Fit a model for an ML/DL strategy and return it with its hyperparameters.

    The target of every model is the next-day change of ``Adj.Close``.

    Args:
        strategy_name: Registered strategy name.
        hyperparams: Must contain ``past_history`` plus the model specific
            keys (``epochs``, ``batch_size`` for LSTM; ``n_estimators``, ...
            for the others).
        train_data: Training OHLCV DataFrame.
        test_data: Test OHLCV DataFrame (used as validation set).

    Returns:
        Tuple ``(model, hyperparams)``; the model is fitted.

    Raises:
        NotImplementedError: For technical-indicator strategies.
    """
    past_history = int(hyperparams["past_history"])

    if strategy_name == "lstm":
        X_train, y_train, X_test, y_test = prepare_data_for_dl(train_data, test_data, past_history)
        model = build_model(strategy_name, hyperparams)
        model.fit(
            X_train,
            y_train,
            epochs=int(hyperparams.get("epochs", 30)),
            batch_size=int(hyperparams.get("batch_size", 32)),
            validation_data=(X_test, y_test),
        )
        return model, hyperparams

    X_train, y_train, X_test, y_test = prepare_data_for_ml(train_data, test_data, past_history)
    model = build_model(strategy_name, hyperparams)
    model.fit(X_train, y_train, validation_data=(X_test, y_test))
    return model, hyperparams


def save_artifacts(
    symbol: str,
    strategy_name: str,
    hyperparams: dict[str, Any],
    model: Any | None,
    artifacts_dir: str | Path = "artifacts",
) -> ArtifactPaths:
    """Persist hyperparameters (XML) and, optionally, the trained model.

    Args:
        symbol: Ticker symbol.
        strategy_name: Registered strategy name.
        hyperparams: Best hyperparameters (written as XML).
        model: Fitted model, or ``None`` for technical strategies.
        artifacts_dir: Root artifacts directory.

    Returns:
        The :class:`ArtifactPaths` that were written.

    Raises:
        ValueError: If a model is provided for a technical-indicator strategy
            or no model is provided for an ML/DL strategy.
    """
    strategy_name = strategy_name.lower()
    strategy_type = STRATEGY_REGISTRY[strategy_name]["type"]
    paths = artifact_paths(symbol, strategy_name, artifacts_dir)

    # ----- Hyperparameters XML -----
    paths.hyperparams.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("Hyperparameters")
    ET.SubElement(root, "strategy_name").text = strategy_name
    for key, value in hyperparams.items():
        element = ET.SubElement(root, key)
        element.text = str(value)
        element.set("type", type(value).__name__)
    ET.ElementTree(root).write(paths.hyperparams, encoding="utf-8", xml_declaration=True)
    logger.info("Hyperparameters saved: %s", paths.hyperparams)

    # ----- Model -----
    if model is not None:
        if strategy_type == StrategyType.TECHNICAL_INDICATOR:
            raise ValueError("Technical strategies do not use models.")
        if strategy_type == StrategyType.DEEP_LEARNING:
            paths.model = paths.model.with_suffix(".h5")
            model.save_model(paths.model)
        else:
            paths.model = paths.model.with_suffix(".pkl")
            model.save_model(paths.model)
    elif strategy_type != StrategyType.TECHNICAL_INDICATOR:
        raise ValueError(f"Strategy '{strategy_name}' requires a trained model.")

    return paths


def load_trained_model(symbol: str, strategy_name: str, hyperparams: dict[str, Any], artifacts_dir: str | Path) -> Any:
    """Load a previously trained model for an ML/DL strategy.

    Args:
        symbol: Ticker symbol.
        strategy_name: Registered strategy name.
        hyperparams: Hyperparameters used to instantiate the wrapper (the
            underlying weights/estimator are replaced by the loaded ones).
        artifacts_dir: Root artifacts directory.

    Returns:
        The fitted model wrapper.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    paths = artifact_paths(symbol, strategy_name.lower(), artifacts_dir)
    if strategy_name.lower() == "lstm":
        paths.model = paths.model.with_suffix(".h5")
        from dl_trading.models.lstm_model import LSTMModel

        wrapper = LSTMModel(
            input_shape=(2, hyperparams["past_history"]),
            lstm_layer_units=hyperparams["number_units_per_layer"],
            stack_depth=hyperparams["stack_depth"],
        )
        wrapper.load_model(paths.model)
        return wrapper

    paths.model = paths.model.with_suffix(".pkl")
    if not paths.model.is_file():
        raise FileNotFoundError(
            f"Model file not found: {paths.model}. Train the strategy first with "
            f"`dl-trading train --symbol {symbol} --strategy {strategy_name}`."
        )
    model_data = joblib.load(paths.model)
    wrapper = build_model(strategy_name.lower(), hyperparams)
    if isinstance(model_data, dict) and "model" in model_data:
        wrapper.model = model_data["model"]
        wrapper.is_fitted = bool(model_data.get("is_fitted", True))
    else:  # Raw estimator dumped by older versions.
        wrapper.model = model_data
        wrapper.is_fitted = True
    return wrapper


def build_strategy(
    strategy_name: str,
    symbol: str,
    hyperparams: dict[str, Any],
    artifacts_dir: str | Path = "artifacts",
) -> Strategy:
    """Instantiate a ready-to-use strategy from a hyperparameter dictionary.

    For ML/DL strategies the trained model is loaded from ``artifacts_dir``;
    for technical strategies the strategy is built directly from the
    parameters.

    Args:
        strategy_name: Registered strategy name.
        symbol: Ticker symbol (used to locate the model file).
        hyperparams: Hyperparameters (must include ``past_history`` and the
            strategy specific keys, plus ``stop_loss`` for the backtester).
        artifacts_dir: Root artifacts directory.

    Returns:
        A strategy instance ready for the backtester.

    Raises:
        KeyError: If the strategy is unknown.
        FileNotFoundError: If an ML/DL model file is missing.
    """
    strategy_name = strategy_name.lower()
    meta = STRATEGY_REGISTRY[strategy_name]
    strategy_type: StrategyType = meta["type"]
    past_history = int(hyperparams["past_history"])
    strategy_class = load_strategy_class(strategy_name)

    if strategy_type == StrategyType.TECHNICAL_INDICATOR:
        return strategy_class(dict(hyperparams), past_history)

    model = load_trained_model(symbol, strategy_name, hyperparams, artifacts_dir)
    return strategy_class(model, past_history)


def load_strategy_class(strategy_name: str) -> type:
    """Resolve the strategy class of a registered strategy name."""
    from dl_trading.strategies.registry import load_class_from_string

    return load_class_from_string(STRATEGY_REGISTRY[strategy_name]["class"])


def load_or_default_hyperparams(
    symbol: str,
    strategy_name: str,
    artifacts_dir: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the saved hyperparameters of a (symbol, strategy) pair.

    Args:
        symbol: Ticker symbol.
        strategy_name: Registered strategy name.
        artifacts_dir: Root artifacts directory.
        overrides: Keys that take precedence over the file values.

    Returns:
        The merged hyperparameter dictionary.

    Raises:
        FileNotFoundError: If no XML file exists for this pair.
    """
    paths = artifact_paths(symbol, strategy_name.lower(), artifacts_dir)
    hyperparams = load_hyperparameters(paths.hyperparams)
    if overrides:
        hyperparams.update({k: v for k, v in overrides.items() if v is not None})
    return hyperparams


def ensure_rsi_defaults(hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Fill in sensible RSI defaults when parameters are missing.

    Args:
        hyperparams: Hyperparameter dictionary (modified copy is returned).

    Returns:
        The completed dictionary.
    """
    completed = dict(hyperparams)
    completed.setdefault("window", 14)
    completed.setdefault("overbought", 70)
    completed.setdefault("oversold", 30)
    completed.setdefault("past_history", int(completed["window"]) + 1)
    return completed


__all__ = [
    "ArtifactPaths",
    "artifact_paths",
    "build_model",
    "build_strategy",
    "ensure_rsi_defaults",
    "load_or_default_hyperparams",
    "load_strategy_class",
    "load_trained_model",
    "save_artifacts",
    "train_model",
]
