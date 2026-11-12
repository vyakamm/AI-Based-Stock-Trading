"""Tests for dl_trading.data.loader."""

from __future__ import annotations

import pandas as pd
import pytest

from dl_trading.data.loader import (
    REQUIRED_COLUMNS,
    download_and_split_data,
    load_hyperparameters,
    load_local_csv,
)

XML_BODY = """<?xml version='1.0' encoding='utf-8'?>
<Hyperparameters>
  <stop_loss type="float">0.05</stop_loss>
  <past_history type="int">60</past_history>
  <kernel type="str">rbf</kernel>
  <C>1.5</C>
  <n_neighbors>7</n_neighbors>
</Hyperparameters>
"""


def test_load_hyperparameters_types(tmp_path):
    path = tmp_path / "best_hyperparams_TEST_ma.xml"
    path.write_text(XML_BODY)

    params = load_hyperparameters(path)
    assert params["stop_loss"] == 0.05 and isinstance(params["stop_loss"], float)
    assert params["past_history"] == 60 and isinstance(params["past_history"], int)
    assert params["kernel"] == "rbf"
    assert params["C"] == 1.5
    assert params["n_neighbors"] == 7 and isinstance(params["n_neighbors"], int)


def test_load_hyperparameters_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="dl-trading train"):
        load_hyperparameters(tmp_path / "nope.xml")


def test_load_hyperparameters_bad_root(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text("<Something><a>1</a></Something>")
    with pytest.raises(ValueError, match="Hyperparameters"):
        load_hyperparameters(path)


def test_load_local_csv_renames_adj_close(tmp_path):
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=5),
            "Open": range(5),
            "High": range(5),
            "Low": range(5),
            "Close": range(5),
            "Volume": range(5),
            "Adj Close": range(5),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    loaded = load_local_csv(path)
    assert "Adj.Close" in loaded.columns
    assert "Adj Close" not in loaded.columns
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_load_local_csv_missing_columns(tmp_path):
    df = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=3), "Open": range(3)})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required column"):
        load_local_csv(path)


def test_load_local_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_local_csv(tmp_path / "ghost.csv")


def test_download_and_split_data_local(ohlcv, tmp_path):
    path = tmp_path / "AMZN.csv"
    ohlcv.rename_axis("Date").to_csv(path)
    train, test = download_and_split_data("AMZN", train_ratio=0.8, local_path=path)

    assert len(train) + len(test) == len(ohlcv)
    assert len(train) == int(len(ohlcv) * 0.8)
    assert train.index.max() < test.index.min()


def test_download_and_split_data_invalid_ratio(ohlcv, tmp_path):
    path = tmp_path / "AMZN.csv"
    ohlcv.rename_axis("Date").to_csv(path)
    with pytest.raises(ValueError, match="train_ratio"):
        download_and_split_data("AMZN", train_ratio=1.5, local_path=path)


def test_required_columns_tuple():
    assert "Adj.Close" in REQUIRED_COLUMNS
