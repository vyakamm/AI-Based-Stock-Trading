"""Tests for dl_trading.data.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_trading.data.preprocessing import prepare_data_for_dl, prepare_data_for_ml


def test_prepare_data_for_ml_shapes_and_alignment(ohlcv):
    ph = 20
    X_train, y_train, X_test, y_test = prepare_data_for_ml(ohlcv, ohlcv.tail(100), ph)

    assert X_train.shape == (len(ohlcv) - ph, 2 * ph)
    assert y_train.shape == (len(ohlcv) - ph,)
    assert X_test.shape == (100 - ph, 2 * ph)
    assert y_test.shape == (100 - ph,)

    # Causality: y[i] must be the change between day i-1 and day i.
    closes = ohlcv["Adj.Close"].values
    assert np.allclose(y_train, closes[ph:] - closes[ph - 1 : -1])

    # Feature content: window 0 holds (open-adjclose, high-low) for day 0.
    window = X_train[0].reshape(ph, 2)
    assert np.isclose(window[0, 0], ohlcv["Open"].iloc[0] - ohlcv["Adj.Close"].iloc[0])
    assert np.isclose(window[0, 1], ohlcv["High"].iloc[0] - ohlcv["Low"].iloc[0])
    assert np.isclose(window[-1, 0], ohlcv["Open"].iloc[ph - 1] - ohlcv["Adj.Close"].iloc[ph - 1])


def test_prepare_data_for_dl_shapes(ohlcv):
    ph = 30
    X_train, y_train, X_test, y_test = prepare_data_for_dl(ohlcv, ohlcv.tail(100), ph)

    assert X_train.shape == (len(ohlcv) - ph, 2, ph)
    assert X_test.shape == (100 - ph, 2, ph)
    assert len(X_train) == len(y_train)

    # Channel 0 = overnight gap, channel 1 = intraday range.
    closes = ohlcv["Adj.Close"].values
    assert np.allclose(X_train[0, 0], ohlcv["Open"].values[:ph] - closes[:ph])
    assert np.allclose(X_train[0, 1], ohlcv["High"].values[:ph] - ohlcv["Low"].values[:ph])
    assert np.allclose(y_train, closes[ph:] - closes[ph - 1 : -1])


@pytest.mark.parametrize("bad", [0, -5])
def test_prepare_data_invalid_past_history(ohlcv, bad):
    with pytest.raises(ValueError, match="past_history"):
        prepare_data_for_ml(ohlcv, ohlcv, bad)
    with pytest.raises(ValueError, match="past_history"):
        prepare_data_for_dl(ohlcv, ohlcv, bad)


def test_prepare_data_missing_column():
    df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2]})
    with pytest.raises(ValueError, match="Adj.Close"):
        prepare_data_for_ml(df, df, 1)
