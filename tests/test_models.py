"""Tests for model wrappers (sklearn, XGBoost, LSTM)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_ohlcv
from dl_trading.data.preprocessing import prepare_data_for_ml
from dl_trading.models.base import MlModel, TrainingHistory
from dl_trading.models.sklearn_models import (
    KNNModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)

try:
    import xgboost  # noqa: F401

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    import tensorflow  # noqa: F401

    HAS_TF = True
except ImportError:  # pragma: no cover
    HAS_TF = False

requires_xgb = pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
requires_tf = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")


@pytest.fixture(scope="module")
def ml_dataset():
    data = make_ohlcv(n=200, seed=11)
    X_train, y_train, X_test, y_test = prepare_data_for_ml(data, data.tail(50), past_history=10)
    return X_train, y_train, X_test, y_test


MODEL_FACTORIES = {
    "randomforest": lambda: RandomForestModel(n_estimators=10, max_depth=4),
    "svm": lambda: SVMModel(),
    "knn": lambda: KNNModel(n_neighbors=3),
}


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_ml_model_lifecycle(name, ml_dataset, tmp_path):
    X_train, y_train, X_test, y_test = ml_dataset
    model = MODEL_FACTORIES[name]()

    assert isinstance(model, MlModel)
    assert not model.is_fitted
    with pytest.raises(ValueError, match="trained"):
        model.predict(X_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=0)
    assert model.is_fitted
    assert model.history is not None and isinstance(model.history, TrainingHistory)
    assert "val_mse" in model.history.history

    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape

    scalar = model.predict_price(X_test[:1])
    assert isinstance(scalar, float)

    metrics = model.evaluate(X_test, y_test, verbose=0)
    assert {"mse", "mae", "r2"}.issubset(metrics)

    # Persistence roundtrip
    path = tmp_path / f"{name}.pkl"
    model.save_model(path)
    reloaded = MODEL_FACTORIES[name]()
    reloaded.load_model(path)
    assert reloaded.is_fitted
    assert np.allclose(reloaded.predict(X_test), predictions)


def test_randomforest_feature_importances(ml_dataset):
    X_train, y_train, _, _ = ml_dataset
    model = RandomForestModel(n_estimators=10, max_depth=4)
    model.fit(X_train, y_train, verbose=0)
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == X_train.shape[1]
    assert model.get_model_info()["status"] == "fitted"


def test_ml_model_rejects_inconsistent_dimensions(ml_dataset):
    X_train, y_train, _, _ = ml_dataset
    model = RandomForestModel(n_estimators=5)
    with pytest.raises(ValueError, match="Inconsistent"):
        model.fit(X_train, y_train[:-1], verbose=0)


@requires_xgb
def test_xgboost_model(ml_dataset, tmp_path):
    X_train, y_train, X_test, y_test = ml_dataset
    model = XGBoostModel(n_estimators=10, max_depth=3, learning_rate=0.2)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=0)
    assert model.is_fitted
    assert model.predict(X_test).shape == y_test.shape

    path = tmp_path / "xgb.pkl"
    model.save_model(path)
    reloaded = XGBoostModel()
    reloaded.load_model(path)
    assert np.allclose(reloaded.predict(X_test), model.predict(X_test))


def test_save_unfitted_model_raises(tmp_path):
    with pytest.raises(ValueError, match="trained"):
        RandomForestModel().save_model(tmp_path / "m.pkl")


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
@requires_tf
def test_lstm_build_predict_save_load(tmp_path):
    from dl_trading.models.lstm_model import LSTMModel

    past_history, units, depth = 8, 4, 2
    model = LSTMModel(input_shape=(2, past_history), lstm_layer_units=units, stack_depth=depth)

    assert model.model.count_params() > 0
    sample = np.random.default_rng(0).normal(size=(2, past_history))
    prediction = model.predict_price(sample)
    assert isinstance(prediction, float)

    # Batch prediction shape
    batch = np.random.default_rng(1).normal(size=(5, 2, past_history))
    assert model.predict(batch).shape == (5, 1)

    # Save / load roundtrip
    path = tmp_path / "lstm.h5"
    model.save_model(path)
    reloaded = LSTMModel(input_shape=(2, past_history), lstm_layer_units=units, stack_depth=depth)
    reloaded.load_model(path)
    assert np.isclose(reloaded.predict_price(sample), prediction)


@requires_tf
def test_lstm_fit_two_epochs():
    from dl_trading.models.lstm_model import LSTMModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 2, 6))
    y = rng.normal(size=(32,))
    model = LSTMModel(input_shape=(2, 6), lstm_layer_units=4, stack_depth=1)
    model.fit(X, y, epochs=2, batch_size=8, validation_data=(X, y), verbose=0, early_stopping=True, patience=5)

    assert model.history is not None
    assert len(model.history.history["loss"]) == 2
