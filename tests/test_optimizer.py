"""Tests for the Optuna optimizer."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from conftest import make_ohlcv
from dl_trading.optimization.optimizer import StrategyOptimizer


def test_optimizer_unknown_strategy():
    data = make_ohlcv(n=80)
    with pytest.raises(KeyError, match="Unknown strategy"):
        StrategyOptimizer("moonshot", "TEST", data, data)


def test_optimizer_technical_strategy(tmp_path):
    """MA optimization needs no training; a few trials must produce artifacts."""
    data = make_ohlcv(n=300, seed=9)

    # Keep the test window well above the maximum sampled long_window (100).
    optimizer = StrategyOptimizer(
        "ma", "TEST", data, data.tail(150), artifacts_dir=tmp_path, stop_loss_range=(0.02, 0.05)
    )
    optimizer.optimize(n_trials=3, seed=0)

    assert optimizer.best_correlation > -1.0

    xml_path = tmp_path / "hyperparams" / "best_hyperparams_TEST_ma.xml"
    assert xml_path.is_file(), "Best hyperparameters must be persisted during optimization"

    root = ET.parse(xml_path).getroot()
    params = {child.tag: child.text for child in root}
    assert "stop_loss" in params
    assert "short_window" in params and "long_window" in params
    assert "strategy_name" in params and params["strategy_name"] == "ma"
