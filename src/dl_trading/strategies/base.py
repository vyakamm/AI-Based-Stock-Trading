"""Trading strategy interface and implementations.

A strategy turns a sliding window of market data into a
:class:`TradingSignal`. Three families are supported:

- **Technical indicators** (moving-average crossover, RSI) - rule based,
  no training required.
- **Classical machine learning** (Random Forest, XGBoost, SVM, KNN) - the
  model predicts the next-day price change from flattened features.
- **Deep learning** (stacked LSTM) - the model consumes the raw
  ``(2, past_history)`` sequence of feature channels.

Every strategy implements :meth:`Strategy.generate_signal`, which makes them
interchangeable inside the :class:`~dl_trading.backtesting.backtester.Backtester`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Actions the backtester understands."""

    HOLD = 0
    BUY = 1
    SELL = -1
    #: Emitted by the risk manager when a stop loss is hit.
    CLOSE = 2


class StrategyType(Enum):
    """Family of a strategy - drives model loading in the pipeline."""

    TECHNICAL_INDICATOR = 1
    MACHINE_LEARNING = 2
    DEEP_LEARNING = 3


class Strategy(ABC):
    """Base class for every trading strategy.

    Attributes:
        strategy_type: Family of the strategy.
        params: Strategy-specific parameters.
        past_history: Number of past days required to emit a signal.
    """

    def __init__(self, strategy_type: StrategyType, params: dict[str, Any], past_history: int) -> None:
        self.strategy_type = strategy_type
        self.params = params
        self.past_history = past_history

    @abstractmethod
    def generate_signal(self, data_window: pd.DataFrame) -> TradingSignal:
        """Return the trading signal for the current step.

        Args:
            data_window: DataFrame with the last ``past_history`` rows of
                market data (Open / High / Low / Adj.Close columns).

        Returns:
            The signal to execute.
        """
        raise NotImplementedError


class MovingAverageCrossoverStrategy(Strategy):
    """Buy when the short SMA crosses above the long SMA, sell otherwise.

    Required params:
        short_window: Period of the fast simple moving average.
        long_window: Period of the slow simple moving average.
    """

    def __init__(self, params: dict[str, Any], past_history: int) -> None:
        super().__init__(StrategyType.TECHNICAL_INDICATOR, params, past_history)

    def generate_signal(self, data_window: pd.DataFrame) -> TradingSignal:
        short_ma = data_window["Adj.Close"].rolling(window=self.params["short_window"]).mean().iloc[-1]
        long_ma = data_window["Adj.Close"].rolling(window=self.params["long_window"]).mean().iloc[-1]

        if np.isnan(short_ma) or np.isnan(long_ma):
            return TradingSignal.HOLD
        if short_ma > long_ma:
            return TradingSignal.BUY
        if short_ma < long_ma:
            return TradingSignal.SELL
        return TradingSignal.HOLD


class RSIStrategy(Strategy):
    """Buy when the RSI is oversold, sell when it is overbought.

    Required params:
        window: RSI look-back period (Wilder smoothing).
        oversold: RSI level below which the asset is considered oversold.
        overbought: RSI level above which the asset is considered overbought.
    """

    def __init__(self, params: dict[str, Any], past_history: int) -> None:
        super().__init__(StrategyType.TECHNICAL_INDICATOR, params, past_history)

    @staticmethod
    def rsi(series: pd.Series, window: int) -> float:
        """Compute the last Relative Strength Index value of a price series.

        Uses Wilder's smoothing (exponential moving average with
        ``alpha = 1 / window``), the convention used by most charting tools.

        Args:
            series: Price series (last value is the latest close).
            window: RSI look-back period.

        Returns:
            The RSI value in the ``[0, 100]`` range.
        """
        delta = series.diff().dropna()
        if len(delta) < window:
            return float("nan")

        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean().iloc[-1]
        avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean().iloc[-1]

        if avg_loss == 0:
            # No loss at all: all-gain -> 100, perfectly flat -> neutral 50.
            return 100.0 if avg_gain > 0 else 50.0
        relative_strength = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + relative_strength))

    def generate_signal(self, data_window: pd.DataFrame) -> TradingSignal:
        rsi_value = self.rsi(data_window["Adj.Close"], int(self.params["window"]))
        if np.isnan(rsi_value):
            return TradingSignal.HOLD

        if rsi_value <= self.params["oversold"]:
            return TradingSignal.BUY
        if rsi_value >= self.params["overbought"]:
            return TradingSignal.SELL
        return TradingSignal.HOLD


class MLStrategyBase(Strategy, ABC):
    """Common machinery for strategies driven by a fitted ML model.

    The model receives a flattened feature vector: ``(Open - Adj.Close)`` and
    ``(High - Low)`` for each of the ``past_history`` days. A positive
    predicted next-day price change yields a BUY signal, otherwise SELL.
    """

    def __init__(self, model: Any, past_history: int) -> None:
        super().__init__(StrategyType.MACHINE_LEARNING, {"model": model}, past_history)
        self.model = model

    def _extract_features(self, data_window: pd.DataFrame) -> np.ndarray:
        """Flatten a window into the ``(1, 2 * past_history)`` feature vector.

        Args:
            data_window: DataFrame with exactly ``past_history`` rows.

        Returns:
            2D feature array with a single row.

        Raises:
            ValueError: If the flattened dimension is unexpected.
        """
        features: list[float] = []
        for i in range(len(data_window)):
            overnight_gap = data_window["Open"].iloc[i] - data_window["Adj.Close"].iloc[i]
            daily_range = data_window["High"].iloc[i] - data_window["Low"].iloc[i]
            features.extend([overnight_gap, daily_range])

        feature_array = np.asarray(features, dtype=np.float64).reshape(1, -1)
        expected_features = self.past_history * 2
        if feature_array.shape[1] != expected_features:
            raise ValueError(
                f"Incorrect feature dimension: got {feature_array.shape[1]}, "
                f"expected {expected_features} (past_history={self.past_history})."
            )
        return feature_array

    def generate_signal(self, data_window: pd.DataFrame) -> TradingSignal:
        try:
            if len(data_window) < self.past_history:
                return TradingSignal.HOLD

            window = data_window.iloc[-self.past_history :]
            features = self._extract_features(window)
            prediction = float(self.model.predict_price(features))

            return TradingSignal.BUY if prediction > 0 else TradingSignal.SELL
        except Exception:  # noqa: BLE001 - a strategy must never crash a backtest
            logger.exception("Signal generation failed; falling back to HOLD")
            return TradingSignal.HOLD


class RandomForestStrategy(MLStrategyBase):
    """Strategy backed by a ``RandomForestModel``."""

    def __init__(self, model: Any, past_history: int) -> None:
        super().__init__(model, past_history)


class XGBoostStrategy(MLStrategyBase):
    """Strategy backed by an ``XGBoostModel``."""

    def __init__(self, model: Any, past_history: int) -> None:
        super().__init__(model, past_history)


class SVMStrategy(MLStrategyBase):
    """Strategy backed by an ``SVMModel``."""

    def __init__(self, model: Any, past_history: int) -> None:
        super().__init__(model, past_history)


class KNNStrategy(MLStrategyBase):
    """Strategy backed by a ``KNNModel``."""

    def __init__(self, model: Any, past_history: int) -> None:
        super().__init__(model, past_history)


class LSTMStrategy(Strategy):
    """Strategy backed by a stacked LSTM operating on raw sequences.

    The model receives the ``(2, past_history)`` array of the two feature
    channels (overnight gap and intraday range) and predicts the next-day
    price change.
    """

    def __init__(self, model: Any, past_history: int) -> None:
        super().__init__(StrategyType.DEEP_LEARNING, {"model": model}, past_history)
        self.model = model

    def generate_signal(self, data_window: pd.DataFrame) -> TradingSignal:
        open_close_diff = (data_window["Open"] - data_window["Adj.Close"]).values
        high_low_diff = (data_window["High"] - data_window["Low"]).values
        input_data = np.asarray([open_close_diff, high_low_diff], dtype=np.float64)
        prediction = self.model.predict_price(input_data)
        return TradingSignal.BUY if prediction > 0 else TradingSignal.SELL
