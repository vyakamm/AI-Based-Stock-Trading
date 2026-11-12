"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 300,
    seed: int = 42,
    start: str = "2022-01-03",
    drift: float = 0.0005,
    vol: float = 0.02,
) -> pd.DataFrame:
    """Generate a synthetic daily OHLCV random walk.

    Args:
        n: Number of business days.
        seed: RNG seed.
        start: First date.
        drift: Daily log-return drift.
        vol: Daily log-return volatility.

    Returns:
        DataFrame with Open / High / Low / Close / Volume / Adj.Close columns
        and a business-day DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume, "Adj.Close": close},
        index=index,
    )


class DummyMLModel:
    """Minimal model double used to test ML strategies deterministically."""

    def __init__(self, value: float = 1.0, fail: bool = False) -> None:
        self.value = value
        self.fail = fail
        self.last_input = None

    def predict_price(self, input_data):
        if self.fail:
            raise RuntimeError("boom")
        self.last_input = np.asarray(input_data)
        return np.array([self.value])


class DummyLSTMModel:
    """Minimal model double for LSTM strategies; records the input shape."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.last_shape = None

    def predict_price(self, input_data):
        self.last_shape = np.asarray(input_data).shape
        return self.value


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def dummy_ml_model() -> DummyMLModel:
    return DummyMLModel()


@pytest.fixture
def dummy_lstm_model() -> DummyLSTMModel:
    return DummyLSTMModel()
