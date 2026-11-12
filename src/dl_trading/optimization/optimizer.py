"""Hyperparameter optimization with Optuna.

:class:`StrategyOptimizer` searches the hyperparameter space defined in the
strategy registry. The objective maximizes the **Pearson correlation between
time and portfolio value** — i.e. it favors parameter sets whose equity curve
grows steadily over the test window.

For every trial the workflow is:

1. sample ``stop_loss`` plus the strategy-specific parameters,
2. (ML/DL) build, fit and score a fresh model on the train/test split,
3. run a backtest and compute the objective,
4. if the trial improves the best correlation so far, persist the
   hyperparameters and the model via
   :func:`dl_trading.pipeline.save_artifacts`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from scipy.stats import pearsonr

from dl_trading.backtesting.backtester import Backtester, BacktesterParameters
from dl_trading.data.preprocessing import prepare_data_for_dl, prepare_data_for_ml
from dl_trading.pipeline import build_model, save_artifacts
from dl_trading.strategies.base import StrategyType
from dl_trading.strategies.registry import STRATEGY_REGISTRY, load_class_from_string

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Optimize the hyperparameters of a single strategy.

    Args:
        strategy_name: Registered strategy name.
        symbol: Ticker symbol (used for artifact names).
        train_data: Training OHLCV DataFrame.
        test_data: Test OHLCV DataFrame (backtest window for the objective).
        artifacts_dir: Root directory where best artifacts are persisted.
        stop_loss_range: ``(low, high)`` for the sampled stop loss.
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        train_data,
        test_data,
        artifacts_dir: str = "artifacts",
        stop_loss_range: tuple = (0.01, 0.10),
    ) -> None:
        strategy_name = strategy_name.lower()
        if strategy_name not in STRATEGY_REGISTRY:
            raise KeyError(f"Unknown strategy '{strategy_name}'. Available: {sorted(STRATEGY_REGISTRY)}")

        self.strategy_name = strategy_name
        self.symbol = symbol
        self.train_data = train_data
        self.test_data = test_data
        self.artifacts_dir = artifacts_dir
        self.stop_loss_range = stop_loss_range
        self.strategy_info = STRATEGY_REGISTRY[strategy_name]
        self.strategy_type: StrategyType = self.strategy_info["type"]
        self.best_correlation = -np.inf

    # ------------------------------------------------------------------
    def _build_strategy(self, hyperparams: dict[str, Any]):
        """Build the strategy object for the given sampled hyperparameters."""
        strategy_class = load_class_from_string(self.strategy_info["class"])
        past_history = int(hyperparams["past_history"])

        if self.strategy_type == StrategyType.TECHNICAL_INDICATOR:
            return strategy_class(dict(hyperparams), past_history)

        if self.strategy_type == StrategyType.MACHINE_LEARNING:
            X_train, y_train, X_test, y_test = prepare_data_for_ml(self.train_data, self.test_data, past_history)
            model = build_model(self.strategy_name, hyperparams)
            model.fit(X_train, y_train, validation_data=(X_test, y_test))
            return strategy_class(model, past_history)

        # Deep learning
        X_train, y_train, X_test, y_test = prepare_data_for_dl(self.train_data, self.test_data, past_history)
        model = build_model(self.strategy_name, hyperparams)
        model.fit(
            X_train,
            y_train,
            epochs=int(hyperparams["epochs"]),
            batch_size=int(hyperparams["batch_size"]),
            validation_data=(X_test, y_test),
            verbose=0,
        )
        return strategy_class(model, past_history)

    # ------------------------------------------------------------------
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: negative Pearson correlation of equity vs time."""
        hyperparams: dict[str, Any] = {
            "stop_loss": trial.suggest_float("stop_loss", *self.stop_loss_range),
            **self.strategy_info["suggest_params"](trial),
        }

        strategy = self._build_strategy(hyperparams)

        backtest_params = BacktesterParameters(
            stop_loss=hyperparams["stop_loss"],
            past_history=int(hyperparams.get("past_history", 0)),
        )
        backtester = Backtester(
            backtest_params, strategy_name=self.strategy_name, symbol=self.symbol, enable_max_drawdown=False
        )
        try:
            backtester.run_backtest(self.test_data, strategy)
        except ValueError as error:
            # Not enough data for the sampled warm-up window: neutral trial.
            logger.debug("Trial pruned: %s", error)
            return 0.0

        pnl = np.asarray(backtester.portfolio_value)
        if len(pnl) < 3 or np.std(pnl) == 0:
            return 0.0
        correlation, _ = pearsonr(np.arange(len(pnl)), pnl)

        if correlation > self.best_correlation:
            self.best_correlation = correlation
            model = getattr(strategy, "model", None)
            if self.strategy_type == StrategyType.TECHNICAL_INDICATOR:
                model = None
            save_artifacts(self.symbol, self.strategy_name, hyperparams, model, self.artifacts_dir)

        return -correlation

    # ------------------------------------------------------------------
    def optimize(self, n_trials: int = 100, n_jobs: int = 1, seed: int | None = 42) -> dict[str, Any]:
        """Run the Optuna study.

        Args:
            n_trials: Number of trials.
            n_jobs: Parallel trials. Keep ``1`` when optimizing deep learning
                strategies (TensorFlow training is not thread-safe).
            seed: Sampler seed for reproducibility (``None`` = random).

        Returns:
            The best hyperparameter dictionary found.
        """
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)

        logger.info("Best correlation: %.4f", self.best_correlation)
        logger.info("Best params: %s", study.best_params)
        return study.best_params
