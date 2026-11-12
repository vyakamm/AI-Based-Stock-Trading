"""Market data loading, validation and train/test splitting.

This module centralizes every data-access concern of the project:

- reading daily OHLCV data from a local CSV file (the default workflow),
- downloading the same schema from Yahoo Finance,
- validating that all columns required by the strategies are present,
- splitting the history into a train and a test segment,
- reading back the best hyperparameters saved by the optimizer (XML).

The canonical column schema is::

    Date, Open, High, Low, Close, Volume, Adj.Close

``Date`` is used as the DataFrame index. If a ``Adj Close`` column (Yahoo
Finance spelling) is found instead of ``Adj.Close``, it is renamed
automatically.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns that every strategy and the backtester expect to find.
REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Adj.Close")


def _validate_columns(data: pd.DataFrame, source: str) -> None:
    """Raise a descriptive error if required columns are missing.

    Args:
        data: DataFrame to validate.
        source: Human readable name of the data source (for error messages).

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in {source}. "
            f"Expected schema: Date (index), {', '.join(REQUIRED_COLUMNS)}, "
            "Close, Volume (optional)."
        )


def load_local_csv(path: str | Path) -> pd.DataFrame:
    """Read a local CSV file with the canonical OHLCV schema.

    Args:
        path: Path to the CSV file. The file must contain a ``Date`` column
            and at least the :data:`REQUIRED_COLUMNS` columns.

    Returns:
        DataFrame indexed by a sorted, datetime ``Date`` index.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: '{path}'. "
            "Either pass --use-yfinance to download data, or provide a CSV "
            "with columns: Date, Open, High, Low, Close, Volume, Adj.Close."
        )

    logger.info("Reading local data from %s", path)
    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    # Accept the Yahoo Finance spelling and normalize it.
    if "Adj Close" in data.columns and "Adj.Close" not in data.columns:
        data = data.rename(columns={"Adj Close": "Adj.Close"})

    _validate_columns(data, str(path))

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def download_from_yahoo(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Download daily data for ``symbol`` from Yahoo Finance.

    Args:
        symbol: Ticker symbol, e.g. ``"AMZN"`` or ``"EURUSD=X"``.
        start_date: First day to download (``YYYY-MM-DD``).
        end_date: Last day to download (``YYYY-MM-DD``). Defaults to tomorrow.

    Returns:
        DataFrame indexed by date with the canonical OHLCV schema.

    Raises:
        ImportError: If ``yfinance`` is not installed.
        ValueError: If the download returned no data.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError("yfinance is required to download data. Install it with: pip install yfinance") from exc

    if end_date is None:
        end_date = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Downloading %s from Yahoo Finance (%s -> %s)", symbol, start_date, end_date)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)

    if data is None or data.empty:
        raise ValueError(f"No data downloaded for symbol '{symbol}'.")

    # yfinance may return a MultiIndex column header (level 'Ticker').
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(level="Ticker", axis=1)

    if "Adj Close" in data.columns and "Adj.Close" not in data.columns:
        data = data.rename(columns={"Adj Close": "Adj.Close"})

    _validate_columns(data, f"Yahoo Finance ({symbol})")

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def download_and_split_data(
    symbol: str,
    start_date: str = "2020-01-01",
    train_ratio: float = 0.8,
    use_yfinance: bool = False,
    local_path: str | Path = "data/AMZN.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load market data (local file or Yahoo Finance) and split it chronologically.

    Args:
        symbol: Ticker symbol, used for logging and error messages.
        start_date: Start date when downloading (``YYYY-MM-DD``).
        train_ratio: Fraction of the history assigned to the train split
            (strictly between 0 and 1). The split respects chronological
            order, i.e. the test set always comes *after* the train set.
        use_yfinance: If True, download from Yahoo Finance; otherwise read
            ``local_path`` from disk.
        local_path: Path of the local CSV file used when ``use_yfinance`` is
            False.

    Returns:
        Tuple ``(train_data, test_data)`` of DataFrames.

    Raises:
        ValueError: If ``train_ratio`` is not in (0, 1) or the data is invalid.
        FileNotFoundError: If a local file is requested but missing.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}.")

    if use_yfinance:
        data = download_from_yahoo(symbol, start_date)
    else:
        data = load_local_csv(local_path)

    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    logger.info(
        "Split %d rows for %s: %d train rows, %d test rows",
        len(data),
        symbol,
        len(train_data),
        len(test_data),
    )
    return train_data, test_data


def load_hyperparameters(path: str | Path) -> dict[str, Any]:
    """Read best hyperparameters from the XML file written by the optimizer.

    The file has a flat structure::

        <Hyperparameters>
            <stop_loss type="float">0.05</stop_loss>
            <past_history type="int">60</past_history>
            <kernel type="str">rbf</kernel>
        </Hyperparameters>

    Values are converted to ``int`` / ``float`` / ``str`` based on the
    ``type`` attribute when present, falling back to content inference.

    Args:
        path: Path to the XML hyperparameter file.

    Returns:
        Dictionary mapping parameter names to typed values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the XML root element is not ``Hyperparameters``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hyperparameter file not found: '{path}'. "
            "Train the strategy first, e.g. `dl-trading train ...` or "
            "`dl-trading optimize ...`."
        )

    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag.lower() != "hyperparameters":
        raise ValueError(f"Unexpected XML root element '{root.tag}' in {path} (expected 'Hyperparameters').")

    hyperparameters: dict[str, Any] = {}
    for param in root:
        raw_value = param.text
        declared_type = param.get("type")
        if raw_value is None:
            continue
        raw_value = raw_value.strip()
        try:
            if declared_type == "int":
                hyperparameters[param.tag] = int(float(raw_value))
            elif declared_type == "float":
                hyperparameters[param.tag] = float(raw_value)
            elif declared_type == "bool":
                hyperparameters[param.tag] = raw_value.lower() == "true"
            elif declared_type == "str":
                hyperparameters[param.tag] = raw_value
            elif "." in raw_value:
                hyperparameters[param.tag] = float(raw_value)
            elif raw_value.lstrip("-").isdigit():
                hyperparameters[param.tag] = int(raw_value)
            else:
                hyperparameters[param.tag] = raw_value
        except ValueError:
            hyperparameters[param.tag] = raw_value

    logger.debug("Loaded hyperparameters from %s: %s", path, hyperparameters)
    return hyperparameters
