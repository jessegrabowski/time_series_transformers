import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.testtools import extract_value_columns
from time_series_transformers import (
    HamiltonFilterTransformer,
    HPFilterDetrend,
    TimeSeriesStandardScaler,
)


def _two_datetime_axes(backend):
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    columns = {"other": dates, "a": np.arange(5.0)}
    if backend == "pandas":
        return pd.DataFrame(columns, index=pd.Index(dates, name="date"))
    return pl.DataFrame({"date": dates, **columns})


def test_no_datetime_axis_raises(backend):
    indexless = (
        pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        if backend == "pandas"
        else pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    )
    with pytest.raises(ValueError, match="No datetime column or index"):
        TimeSeriesStandardScaler().fit(indexless)


def test_multiple_datetime_axes_raises(backend):
    with pytest.raises(ValueError, match="exactly one datetime axis"):
        TimeSeriesStandardScaler().fit(_two_datetime_axes(backend))


def test_explicit_date_column_does_not_bypass_multiple_axes(backend):
    with pytest.raises(ValueError, match="exactly one datetime axis"):
        TimeSeriesStandardScaler(date_column="date").fit(_two_datetime_axes(backend))


def test_explicit_date_column_is_honored(backend):
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    columns = {"date": dates, "a": np.arange(5.0)}
    X = pd.DataFrame(columns) if backend == "pandas" else pl.DataFrame(columns)
    tf = TimeSeriesStandardScaler(date_column="date").fit(X)
    assert tf.date_column_ == "date"


def test_unknown_date_column_raises(dated_random_walk):
    with pytest.raises(ValueError, match="date_column"):
        TimeSeriesStandardScaler(date_column="nope").fit(dated_random_walk)


def test_transform_preserves_date_axis(dated_random_walk, backend):
    X = dated_random_walk
    out = TimeSeriesStandardScaler().fit(X).transform(X)
    out_dates = out.index.to_numpy() if backend == "pandas" else out["date"].to_numpy()
    in_dates = X.index.to_numpy() if backend == "pandas" else X["date"].to_numpy()
    np.testing.assert_array_equal(out_dates, in_dates)


@pytest.mark.parametrize(
    "transformer",
    [HPFilterDetrend(lamb=1600), HamiltonFilterTransformer(h=2, p=1)],
    ids=["hp", "hamilton"],
)
def test_inverse_aligns_stored_trend_by_date(make_dated, rng, backend, transformer):
    a = np.cumsum(rng.standard_normal(30))
    X = make_dated({"a": a})
    tf = transformer.fit(X)
    cycle = tf.transform(X)
    reversed_cycle = cycle.iloc[::-1] if backend == "pandas" else cycle.reverse()
    recovered = extract_value_columns(tf.inverse_transform(reversed_cycle))["a"]
    finite = np.isfinite(recovered)
    np.testing.assert_allclose(recovered[finite], a[::-1][finite], atol=1e-8)
