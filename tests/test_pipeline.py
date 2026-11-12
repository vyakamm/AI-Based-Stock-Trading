"""Tests for the artifact pipeline (save / load / build strategy)."""

from __future__ import annotations

import pytest

from conftest import make_ohlcv
from dl_trading.data.preprocessing import prepare_data_for_ml
from dl_trading.pipeline import (
    artifact_paths,
    build_model,
    build_strategy,
    ensure_rsi_defaults,
    load_or_default_hyperparams,
    load_trained_model,
    save_artifacts,
    train_model,
)
from dl_trading.strategies.base import (
    MovingAverageCrossoverStrategy,
    RandomForestStrategy,
    RSIStrategy,
    StrategyType,
)


# ---------------------------------------------------------------------------
# artifact_paths
# ---------------------------------------------------------------------------
def test_artifact_paths_naming():
    paths = artifact_paths("AMZN", "lstm", artifacts_dir="artifacts")
    assert paths.hyperparams.name == "best_hyperparams_AMZN_lstm.xml"
    assert paths.model.name == "model_AMZN_lstm"
    assert paths.hyperparams.parent.name == "hyperparams"
    assert paths.model.parent.name == "models"


def test_artifact_paths_unknown_strategy():
    with pytest.raises(KeyError, match="Unknown strategy"):
        artifact_paths("AMZN", "moonshot")


# ---------------------------------------------------------------------------
# build_model / train_model
# ---------------------------------------------------------------------------
def test_build_model_factory():
    from dl_trading.models.lstm_model import LSTMModel
    from dl_trading.models.sklearn_models import RandomForestModel

    assert isinstance(
        build_model("lstm", {"past_history": 20, "number_units_per_layer": 8, "stack_depth": 1}), LSTMModel
    )
    assert isinstance(build_model("randomforest", {"n_estimators": 10, "max_depth": 3}), RandomForestModel)


def test_build_model_rejects_technical_strategy():
    with pytest.raises(NotImplementedError):
        build_model("ma", {})


def test_train_model_randomforest_roundtrip(tmp_path):
    data = make_ohlcv(n=180, seed=5)
    hyperparams = {"past_history": 10, "n_estimators": 10, "max_depth": 4, "stop_loss": 0.03}

    model, hyperparams = train_model("randomforest", hyperparams, data, data.tail(40))
    assert model.is_fitted

    paths = save_artifacts("TEST", "randomforest", hyperparams, model, tmp_path)
    assert paths.hyperparams.is_file() and paths.model.is_file()
    assert paths.model.suffix == ".pkl"

    reloaded = load_trained_model("TEST", "randomforest", hyperparams, tmp_path)
    X, _, _, _ = prepare_data_for_ml(data, data.tail(40), 10)
    assert reloaded.is_fitted


def test_train_model_rejects_technical_strategy(tmp_path):
    data = make_ohlcv(n=60)
    with pytest.raises((NotImplementedError, KeyError)):
        train_model("ma", {"past_history": 5}, data, data)


# ---------------------------------------------------------------------------
# build_strategy
# ---------------------------------------------------------------------------
def test_build_strategy_ma_from_params():
    hyperparams = {
        "short_window": 5,
        "long_window": 20,
        "past_history": 20,
        "stop_loss": 0.03,
    }
    strategy = build_strategy("ma", "TEST", hyperparams, artifacts_dir="artifacts")
    assert isinstance(strategy, MovingAverageCrossoverStrategy)
    assert strategy.past_history == 20


def test_build_strategy_rsi():
    hyperparams = {"window": 14, "overbought": 70, "oversold": 30, "past_history": 15}
    strategy = build_strategy("rsi", "TEST", hyperparams, artifacts_dir="artifacts")
    assert isinstance(strategy, RSIStrategy)


def test_build_strategy_ml_requires_trained_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_strategy(
            "randomforest",
            "TEST",
            {"past_history": 10, "n_estimators": 5, "max_depth": 2},
            artifacts_dir=tmp_path,
        )


def test_build_strategy_ml_roundtrip(tmp_path):
    data = make_ohlcv(n=180, seed=6)
    hyperparams = {"past_history": 10, "n_estimators": 10, "max_depth": 4, "stop_loss": 0.03}

    model, hyperparams = train_model("randomforest", hyperparams, data, data.tail(40))
    save_artifacts("TEST", "randomforest", hyperparams, model, tmp_path)

    strategy = build_strategy("randomforest", "TEST", hyperparams, artifacts_dir=tmp_path)
    assert isinstance(strategy, RandomForestStrategy)
    assert strategy.strategy_type == StrategyType.MACHINE_LEARNING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_load_or_default_hyperparams_overrides(tmp_path):
    (tmp_path / "hyperparams").mkdir()
    xml = tmp_path / "hyperparams" / "best_hyperparams_TEST_ma.xml"
    xml.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<Hyperparameters><short_window type='int'>5</short_window>"
        "<long_window type='int'>20</long_window>"
        "<past_history type='int'>20</past_history></Hyperparameters>"
    )

    params = load_or_default_hyperparams("TEST", "ma", tmp_path, overrides={"short_window": 7})
    assert params["short_window"] == 7
    assert params["long_window"] == 20


def test_ensure_rsi_defaults():
    completed = ensure_rsi_defaults({"window": 10})
    assert completed["overbought"] == 70
    assert completed["oversold"] == 30
    assert completed["past_history"] == 11
