"""Feature/target preparation for machine learning and deep learning models.

Both helpers share the same feature definition:

- ``volatility   = Open - Adj.Close`` (overnight gap proxy)
- ``daily_range  = High - Low``       (intraday range)

For a sample anchored at day ``i`` the model sees the ``past_history`` days
``i - past_history .. i - 1`` and the target is the price change between day
``i - 1`` and day ``i``::

    y[i] = Adj.Close[i] - Adj.Close[i - 1]

This guarantees a strictly causal setup: the features never contain
information posterior to the target window, and the two helpers (ML: flattened
features, DL: sequences) produce perfectly aligned ``(X, y)`` pairs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_columns(data: pd.DataFrame) -> None:
    """Ensure the columns needed to build features are present.

    Args:
        data: OHLCV DataFrame.

    Raises:
        ValueError: If a required column is missing.
    """
    for column in ("Open", "High", "Low", "Adj.Close"):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' is required to build features.")


def prepare_data_for_ml(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    past_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flattened ``(X, y)`` datasets for scikit-learn style models.

    Each sample is a vector of length ``2 * past_history`` alternating
    ``(Open - Adj.Close)`` and ``(High - Low)`` for every day of the window.

    Args:
        train_data: Training OHLCV DataFrame.
        test_data: Test OHLCV DataFrame.
        past_history: Number of past days in each feature window.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If columns are missing or ``past_history < 1``.
    """
    if past_history < 1:
        raise ValueError(f"past_history must be >= 1, got {past_history}")
    _check_columns(train_data)
    _check_columns(test_data)

    def build_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        open_gap = (data["Open"] - data["Adj.Close"]).values
        high_low = (data["High"] - data["Low"]).values
        closes = data["Adj.Close"].values

        windows = []
        for i in range(past_history, len(data)):
            features = []
            for j in range(past_history):
                k = i - past_history + j
                features.extend([open_gap[k], high_low[k]])
            windows.append(features)

        X = np.asarray(windows, dtype=np.float64)
        # Target: change between day (i-1) and day i, for i in [past_history, len-1].
        # Window k (rows [k, k + past_history)) is paired with closes[k + past_history] - closes[k + past_history - 1].
        y = closes[past_history:] - closes[past_history - 1 : -1]
        return X, y

    X_train, y_train = build_split(train_data)
    X_test, y_test = build_split(test_data)

    assert len(X_train) == len(y_train), "X_train and y_train sizes differ"
    assert len(X_test) == len(y_test), "X_test and y_test sizes differ"

    logger.debug(
        "ML dataset built: X_train %s, y_train %s, X_test %s, y_test %s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_train, y_train, X_test, y_test


def prepare_data_for_dl(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    past_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sequence ``(X, y)`` datasets for the LSTM model.

    Every sample has shape ``(2, past_history)``: channel 0 is
    ``Open - Adj.Close`` and channel 1 is ``High - Low`` over the window. The
    target is the same as in :func:`prepare_data_for_ml` so both model
    families learn the identical task.

    Args:
        train_data: Training OHLCV DataFrame.
        test_data: Test OHLCV DataFrame.
        past_history: Number of past days in each sequence.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)`` where the ``X`` arrays
        have shape ``(n_samples, 2, past_history)``.

    Raises:
        ValueError: If columns are missing or ``past_history < 1``.
    """
    if past_history < 1:
        raise ValueError(f"past_history must be >= 1, got {past_history}")
    _check_columns(train_data)
    _check_columns(test_data)

    def build_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        open_gap = (data["Open"] - data["Adj.Close"]).values
        high_low = (data["High"] - data["Low"]).values
        closes = data["Adj.Close"].values

        X = np.asarray(
            [[open_gap[i - past_history : i], high_low[i - past_history : i]] for i in range(past_history, len(data))],
            dtype=np.float64,
        )
        # Target: change between day (i-1) and day i, for i in [past_history, len-1].
        y = closes[past_history:] - closes[past_history - 1 : -1]
        return X, y

    X_train, y_train = build_split(train_data)
    X_test, y_test = build_split(test_data)

    assert len(X_train) == len(y_train), "X_train and y_train sizes differ"
    assert len(X_test) == len(y_test), "X_test and y_test sizes differ"

    logger.debug(
        "DL dataset built: X_train %s, y_train %s, X_test %s, y_test %s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_train, y_train, X_test, y_test
