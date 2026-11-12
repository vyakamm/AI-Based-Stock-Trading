# Methodology

This document describes the data, the feature/target design, the backtest
engine accounting model, and the metrics computed by the framework. It also
lists the simplifying assumptions you should be aware of before interpreting
any result.

## 1. Data

The bundled dataset (`data/AMZN.csv`) contains AMZN daily bars from
2022-01-03 to 2026-11-12 (1221 trading days) with the canonical schema:

| Column | Meaning |
|---|---|
| `Open`, `High`, `Low`, `Close` | OHLC prices |
| `Volume` | traded volume |
| `Adj.Close` | split/dividend-adjusted close — **the execution price** |

A chronological 80/20 split gives 976 training days (2022-01 → 2025-11) and
245 test days (2025-11 → 2026-11). The split is *not* shuffled: the test set
always comes strictly after the training set, mimicking live deployment.

The optional R script `scripts/import_data.R` refreshes the dataset from
Yahoo Finance via `quantmod`.

## 2. Features and target

For a sample anchored at day `i`, the model observes the window
`i - past_history … i - 1` (it never sees day `i` itself) and predicts the
**next-day change of the adjusted close**:

```
y[i] = Adj.Close[i] - Adj.Close[i - 1]
```

Two feature channels are computed per day:

| Channel | Formula | Interpretation |
|---|---|---|
| overnight gap | `Open - Adj.Close` | overnight volatility proxy |
| intraday range | `High - Low` | intraday volatility proxy |

- **Deep learning** (`prepare_data_for_dl`): sequences of shape
  `(2, past_history)` — two timesteps of `past_history`-dimensional feature
  vectors, consumed directly by the LSTM.
- **Classical ML** (`prepare_data_for_ml`): the same channels flattened into a
  `(2 * past_history,)` vector.

Both helpers produce exactly aligned `(X, y)` pairs with identical targets, so
LSTM and sklearn models learn the same task and their backtests are directly
comparable. A positive prediction generates a BUY signal, otherwise SELL.

## 3. Models

| Strategy | Model | Notes |
|---|---|---|
| `lstm` | stacked LSTM (Keras) | dropout 0.5 per layer, Dense(1) head, Adam + MSE, early stopping |
| `randomforest` | `RandomForestRegressor` | feature importances available |
| `xgboost` | `XGBRegressor` | gradient boosting |
| `svm` | `SVR` (RBF) | support vector regression |
| `knn` | `KNeighborsRegressor` | nearest neighbors |
| `ma` | moving-average crossover | rule based (short SMA vs long SMA) |
| `rsi` | RSI thresholds | rule based (Wilder smoothing) |

The ML artifacts (Random Forest, XGBoost, SVM, KNN, MA) were trained with the
project's own CLI / Optuna runs on the bundled dataset. The LSTM checkpoint
predates the refactoring of this repository: it was produced by the author's
original training script on a similar (but not identical) data pipeline, so
`dl-trading train` will not reproduce it bit-for-bit — see §7. The
authoritative hyperparameters of every artifact live in
`artifacts/hyperparams/best_hyperparams_<SYMBOL>_<STRATEGY>.xml`.

- **LSTM**: stacked Keras checkpoint (provenance above, see §7).
- **Random Forest**: `past_history=20`, 200 trees, `max_depth=8`.
- **XGBoost**: `past_history=20`, 200 rounds, `max_depth=6`, `lr=0.1`.
- **SVM**: `past_history=20`, `C=1.0`, RBF kernel.
- **KNN**: `past_history=20`, `k=5`, uniform weights.
- **MA crossover**: `short_window=10`, `long_window=23` (original Optuna run).

## 4. Backtest engine

The backtester is event-driven and replays the strategy day by day:

1. the strategy observes the last `past_history` rows **before** day `i` and
   emits a signal;
2. open positions are checked against the **stop loss** first — an adverse
   move of more than `stop_loss` (relative) synthesizes a `CLOSE`;
3. the signal is executed at `Adj.Close[i]`;
4. the portfolio is marked to market and appended to the equity curve.

Accounting model:

- fixed position size: `trade_units` (default 50) units per position;
- long opening debits `price × units + cost` from cash; short opening credits
  `price × units - cost` (unlimited shorting assumed);
- transaction cost per trade: `max(units × transaction_cost, min_cost)`
  (defaults: 0.35 % per unit with a $0.35 minimum);
- reversal signals (BUY while short / SELL while long) close the position;
  reopening is *not* done in the same step.

Position sizing is intentionally simple so that strategy differences come
from the signals, not from money management. It also means results scale
linearly with `trade_units`.

## 5. Metrics glossary

| Metric | Definition |
|---|---|
| Total return | `(final equity − initial capital) / initial capital` |
| Annualized return | compound annual growth rate implied by the total return |
| Volatility | standard deviation of daily equity returns × √252 |
| Sharpe ratio | mean daily excess return / std × √252 |
| Sortino ratio | like Sharpe but only downside volatility |
| Max drawdown ($) | deepest peak-to-trough drop of the equity curve |
| Max drawdown (%) | same, relative to the running peak |
| Calmar ratio | annualized return / |max drawdown %| (floored at 1 %, capped at 100) |
| Recovery factor | net profit / |max drawdown $| |
| Win rate | share of closed trades with positive PnL |
| Profit factor | gross profit / gross loss |
| Payoff ratio | average win / average loss |
| VaR (95 %) | 5th percentile of daily returns, scaled to currency (monthly = daily × √21) |
| Risk of ruin | gambler's-ruin approximation from win rate and payoff ratio |
| Positive months | share of calendar months with positive month-end return |

## 6. Known limitations (read before drawing conclusions)

1. **Price-only features.** Two volatility channels and no volume, no returns
   lags, no external data. Predictability from such a small feature set is
   structurally limited — see how the strategy ranking flips between the two
   windows documented in §7.
2. **Single objective, single split.** Hyperparameters come from one
   optimization run on one train/test split; walk-forward validation would
   give more robust estimates (planned, see the README roadmap).
3. **Idealized execution.** Fills at the exact close, no slippage, no
   borrowing fees on shorts, no market impact, unlimited liquidity.
4. **Fixed position size.** 50 units regardless of volatility or capital
   growth; results scale linearly with position size.
5. **Overfitting risk in optimization.** The optimizer maximizes the Pearson
   correlation of the equity curve over the *test* window — the best trial is
   therefore selected on out-of-sample data. Treat optimized results as
   in-sample and re-validate on fresh data.
6. **Survivorship / selection bias.** AMZN is a single large-cap name; nothing
   here generalizes to other universes without re-running the pipeline.
7. **Cost sensitivity.** Every trade pays at least the $0.35 minimum cost, so
   a daily-trading model accumulates costs on every signal (~$41 over 117
   trades for the LSTM on this window, vs ~$13 over 36 trades for the MA
   crossover); slippage, borrowing fees and market impact are not modeled.

## 7. Result variability and reproducibility

The benchmark was first published on a different test window. Both snapshots
below were produced by the same code and the same backtest engine; only the
data window changed.

| Window | Best strategy | Worst strategy | Buy & Hold |
|---|---|---|---|
| Nov 2024 → Oct 2025 (242 test days) | KNN **+16.4 %** (Sharpe 1.02) | RSI −10.8 % | +7.4 % |
| Nov 2025 → Nov 2026 (245 test days) | LSTM **+67.6 %** (Sharpe 4.26) | SVM −19.5 % | +15.5 % |

Three lessons:

1. **Strategy rankings flip between windows.** KNN went from first place to
   below Buy & Hold; the LSTM from last place to first. Any single-window
   backtest measures a market regime at least as much as a strategy — this is
   the strongest argument for the walk-forward validation on the roadmap.
2. **The current LSTM result is an outlier, not a promise.** A Sharpe ratio
   of 4.26 with a 72 % win rate over roughly ten months is far beyond what
   survives out-of-sample revalidation in practice. Treat it as a property of
   this window and of this specific checkpoint, not as an expected return.
3. **Neural training is not reproducible across machines.** The LSTM was
   retrained once on an Apple M2 (Metal GPU): floating-point reductions on a
   GPU do not associate in the same order as on CPU, so even with fixed seeds
   the restored weights differ slightly and the final metrics move. Classical
   ML artifacts (tree ensembles, SVR, k-NN) are reproducible given the same
   library versions.

The shipped LSTM checkpoint was trained before the refactoring of this
repository (author's original training script, similar but not identical
feature pipeline). It is kept because it is the checkpoint the published
results were computed with; retraining with `dl-trading train` yields an
equivalent architecture but not the same weights.
