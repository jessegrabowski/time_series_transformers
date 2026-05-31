import numpy as np
import pandas as pd
import pytest

from tests.testtools import assert_columns_close, extract_value_columns
from time_series_transformers import (
    LogTransformer,
    TimeSeriesMinMaxScaler,
    TimeSeriesStandardScaler,
)


def test_pandas_datetime_index_name_and_freq_preserved():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", name="date")
    X = pd.DataFrame({"a": np.arange(10.0)}, index=idx)
    out = TimeSeriesStandardScaler().fit(X).transform(X)
    assert out.index.name == "date"
    assert out.index.freq == idx.freq


@pytest.mark.parametrize("lam", [0, 0.25, 0.5, 1.0, 2.0])
def test_log_transformer_roundtrip(dated_positive_random_walk, lam):
    X = dated_positive_random_walk
    tf = LogTransformer(lam=lam).fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-10)


def test_log_transformer_known_values(make_dated):
    data = make_dated({"x": [1.0, np.e, np.e**2]})
    result = LogTransformer(lam=0).fit(data).transform(data)
    np.testing.assert_allclose(extract_value_columns(result)["x"], [0.0, 1.0, 2.0], atol=1e-12)


def test_log_transformer_lam1_is_shifted_identity(dated_positive_random_walk):
    X = dated_positive_random_walk
    result = LogTransformer(lam=1.0).fit(X).transform(X)
    assert_columns_close(
        result, {k: v - 1.0 for k, v in extract_value_columns(X).items()}, atol=1e-12
    )


def test_standard_scaler_roundtrip(dated_random_walk):
    X = dated_random_walk
    tf = TimeSeriesStandardScaler().fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-12)


def test_standard_scaler_produces_zero_mean_unit_variance(dated_random_walk):
    result = TimeSeriesStandardScaler().fit(dated_random_walk).transform(dated_random_walk)
    for col in extract_value_columns(result).values():
        np.testing.assert_allclose(col.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(col.std(ddof=1), 1.0, atol=1e-12)


def test_standard_scaler_zero_variance_column_passes_through(make_dated):
    data = make_dated({"const": [5.0] * 10, "vary": [float(i) for i in range(10)]})
    result = TimeSeriesStandardScaler().fit(data).transform(data)
    np.testing.assert_allclose(extract_value_columns(result)["const"], 0.0)


def test_minmax_scaler_roundtrip(dated_random_walk):
    X = dated_random_walk
    tf = TimeSeriesMinMaxScaler().fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-12)


def test_minmax_scaler_produces_01_range(dated_random_walk):
    result = TimeSeriesMinMaxScaler().fit(dated_random_walk).transform(dated_random_walk)
    for col in extract_value_columns(result).values():
        np.testing.assert_allclose(col.min(), 0.0, atol=1e-12)
        np.testing.assert_allclose(col.max(), 1.0, atol=1e-12)


def test_minmax_scaler_constant_column_passes_through(make_dated):
    data = make_dated({"const": [3.0] * 10, "vary": [float(i) for i in range(10)]})
    result = TimeSeriesMinMaxScaler().fit(data).transform(data)
    np.testing.assert_allclose(extract_value_columns(result)["const"], 0.0)
