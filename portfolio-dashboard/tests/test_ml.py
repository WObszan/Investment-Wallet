import numpy as np
import pandas as pd
import pytest

from ml_model import prepare_futures, run_backtest


class FakeModel:
    """Mock model to avoid training real models during unit testing."""
    def __init__(self, prediction: int):
        self.prediction = prediction

    def predict(self, X):
        return np.full(len(X), self.prediction)


def _make_price_series(n=260, seed=1):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    volume = np.random.randint(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Close": price, "Volume": volume}, index=dates)


#prepare_futures
def test_prepare_futures_drops_last_horizon_rows_instead_of_faking_target():
    # Verify that target contains no NaNs and that the last result date is earlier than input
    df = _make_price_series()
    benchmark_df = _make_price_series(seed=2)

    result = prepare_futures(df, benchmark_df)

    assert not result["Target"].isna().any()
    assert set(result["Target"].unique()).issubset({0, 1})
    assert result.index.max() < df.index.max()


#run_backtest
def test_run_backtest_zero_signal_gives_zero_strategy_return():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=0), X_test, y_test, original_data, horizon=5)

    assert result["Strategy_Cumulative"].iloc[-1] == pytest.approx(0.0)


def test_run_backtest_always_buy_matches_buy_and_hold():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=1), X_test, y_test, original_data, horizon=5)

    max_diff = (result["Strategy_Cumulative"] - result["Market_Cumulative"]).abs().max()
    assert max_diff < 1e-9


def test_run_backtest_resamples_non_overlapping_every_horizon_days():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=1), X_test, y_test, original_data, horizon=5)

    # 30 test days / horizon=5 yields 6 non-overlapping evaluation points
    assert len(result) == 6