import numpy as np
import pytest

from tests.testtools import (
    assert_batch_matches_per_series,
    assert_columns_close,
    extract_value_columns,
)
from time_series_transformers import (
    DetrendTransformer,
    DifferenceTransformer,
    HamiltonFilterTransformer,
    HPFilterDetrend,
)


def test_difference_roundtrip(dated_random_walk):
    X = dated_random_walk
    tf = DifferenceTransformer().fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-12)


def test_difference_known_values(make_dated):
    data = make_dated({"x": [10.0, 13.0, 11.0]})
    result = DifferenceTransformer().fit(data).transform(data)
    np.testing.assert_allclose(extract_value_columns(result)["x"][1:], [3.0, -2.0])


@pytest.mark.parametrize("periods", [1, 2, 4, 12])
def test_difference_periods_roundtrip(dated_random_walk, periods):
    X = dated_random_walk
    tf = DifferenceTransformer(periods=periods).fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-10)


def test_difference_periods_known_values(make_dated):
    data = make_dated({"x": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]})
    result = extract_value_columns(DifferenceTransformer(periods=2).fit(data).transform(data))["x"]
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], [3.0, 5.0, 7.0, 9.0])


def test_difference_periods_seasonal_known_values(make_dated):
    data = make_dated({"x": np.arange(8, dtype=float) + 10.0})
    tf = DifferenceTransformer(periods=4).fit(data)
    diffed = tf.transform(data)
    diffed_x = extract_value_columns(diffed)["x"]
    assert np.isnan(diffed_x[:4]).all()
    np.testing.assert_allclose(diffed_x[4:], 4.0)
    assert_columns_close(tf.inverse_transform(diffed), data, atol=1e-12)


def test_difference_invalid_periods_raises(dated_random_walk):
    with pytest.raises(ValueError, match="periods must be >= 1"):
        DifferenceTransformer(periods=0).fit(dated_random_walk)


def test_difference_short_series_raises(make_dated):
    short = make_dated({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least periods=4"):
        DifferenceTransformer(periods=4).fit(short)


@pytest.mark.parametrize("trend", ["c", "t", "ct", "ctt"])
def test_detrend_roundtrip(dated_random_walk, trend):
    X = dated_random_walk
    tf = DetrendTransformer(trend=trend).fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-10)


def test_detrend_constant_removes_mean(dated_random_walk):
    result = DetrendTransformer(trend="c").fit(dated_random_walk).transform(dated_random_walk)
    for col in extract_value_columns(result).values():
        np.testing.assert_allclose(col.mean(), 0.0, atol=1e-10)


def test_detrend_linear_removes_linear_trend(make_dated, rng):
    t = np.arange(100, dtype=float)
    data = make_dated({"x": 3.0 + 2.0 * t + rng.standard_normal(100) * 0.01})
    residuals = DetrendTransformer(trend="ct").fit(data).transform(data)
    assert np.abs(extract_value_columns(residuals)["x"]).max() < 0.1


def test_detrend_invalid_trend_raises(dated_random_walk):
    with pytest.raises(ValueError, match="Invalid trend"):
        DetrendTransformer(trend="xyz").fit(dated_random_walk)


def test_detrend_nan_values_do_not_corrupt_finite_values(make_dated, rng):
    a = np.cumsum(rng.standard_normal(50))
    a[5] = np.nan
    data = make_dated({"a": a, "b": np.cumsum(rng.standard_normal(50))})
    tf = DetrendTransformer(trend="ct").fit(data)
    recovered = extract_value_columns(tf.inverse_transform(tf.transform(data)))
    original = extract_value_columns(data)
    for name in recovered:
        finite = np.isfinite(original[name])
        np.testing.assert_allclose(recovered[name][finite], original[name][finite], atol=1e-10)


def test_detrend_transform_length_independent_of_fit(make_dated, rng):
    train = make_dated({"x": rng.standard_normal(100)})
    test = make_dated({"x": rng.standard_normal(30)})
    tf = DetrendTransformer(trend="ct").fit(train)
    assert_columns_close(tf.inverse_transform(tf.transform(test)), test, atol=1e-10)


@pytest.mark.parametrize("h,p", [(1, 1), (2, 2), (4, 2), (8, 4)])
def test_hamilton_roundtrip(dated_random_walk, h, p):
    X = dated_random_walk
    tf = HamiltonFilterTransformer(h=h, p=p).fit(X)
    recovered = extract_value_columns(tf.inverse_transform(tf.transform(X)))
    original = extract_value_columns(X)
    for name in recovered:
        finite = np.isfinite(recovered[name])
        np.testing.assert_allclose(recovered[name][finite], original[name][finite], atol=1e-10)


def test_hamilton_nan_structure(dated_random_walk):
    h, p = 3, 2
    cycle = HamiltonFilterTransformer(h=h, p=p).fit(dated_random_walk).transform(dated_random_walk)
    n_leading_nan = h + p - 1
    for col in extract_value_columns(cycle).values():
        assert np.isnan(col[:n_leading_nan]).all()
        assert np.isfinite(col[n_leading_nan:]).all()


def test_hamilton_inverse_without_store_trend_raises(dated_random_walk):
    tf = HamiltonFilterTransformer(store_trend=False).fit(dated_random_walk)
    with pytest.raises(ValueError, match="store_trend"):
        tf.inverse_transform(tf.transform(dated_random_walk))


def test_hamilton_short_series_raises(make_dated):
    short = make_dated({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="too short"):
        HamiltonFilterTransformer(h=2, p=2).fit(short)


def test_hp_filter_roundtrip(dated_random_walk):
    X = dated_random_walk
    tf = HPFilterDetrend(lamb=1600).fit(X)
    assert_columns_close(tf.inverse_transform(tf.transform(X)), X, atol=1e-8)


@pytest.mark.parametrize("n", [3, 4, 5, 50])
def test_hp_filter_matches_dense_solve(make_dated, rng, n):
    lamb = 1600.0
    y = rng.standard_normal(n)
    data = make_dated({"x": y})
    cycle = extract_value_columns(HPFilterDetrend(lamb=lamb).fit(data).transform(data))["x"]
    trend = y - cycle

    second_diff = np.diff(np.eye(n), n=2, axis=0)
    dense = np.eye(n) + lamb * (second_diff.T @ second_diff)
    np.testing.assert_allclose(trend, np.linalg.solve(dense, y), atol=1e-8)


def test_hp_filter_linear_series_has_zero_cycle(make_dated):
    t = np.arange(100, dtype=float)
    data = make_dated({"x": 3.0 + 2.0 * t})
    cycle = HPFilterDetrend(lamb=1600).fit(data).transform(data)
    assert np.abs(extract_value_columns(cycle)["x"]).max() < 1e-6


def test_hp_filter_larger_lamb_yields_smoother_trend(make_dated, rng):
    data = make_dated({"a": np.cumsum(rng.standard_normal(50))})
    a = extract_value_columns(data)["a"]
    trend_rough = (
        a - extract_value_columns(HPFilterDetrend(lamb=100).fit(data).transform(data))["a"]
    )
    trend_smooth = (
        a - extract_value_columns(HPFilterDetrend(lamb=1e7).fit(data).transform(data))["a"]
    )
    assert np.abs(np.diff(trend_smooth, 2)).sum() < np.abs(np.diff(trend_rough, 2)).sum()


def test_hp_filter_inverse_without_store_trend_raises(dated_random_walk):
    tf = HPFilterDetrend(store_trend=False).fit(dated_random_walk)
    with pytest.raises(ValueError, match="store_trend"):
        tf.inverse_transform(tf.transform(dated_random_walk))


@pytest.mark.parametrize("lamb", [-1, 0])
def test_hp_filter_invalid_lamb_raises(dated_random_walk, lamb):
    with pytest.raises(ValueError, match="lamb must be positive"):
        HPFilterDetrend(lamb=lamb).fit(dated_random_walk)


def test_hp_filter_short_series_raises(make_dated):
    short = make_dated({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="too short"):
        HPFilterDetrend().fit(short)


def test_hp_filter_nan_values_do_not_corrupt_finite_values(make_dated, rng):
    a = np.cumsum(rng.standard_normal(50))
    a[5] = np.nan
    data = make_dated({"a": a, "b": np.cumsum(rng.standard_normal(50))})
    tf = HPFilterDetrend().fit(data)
    recovered = extract_value_columns(tf.inverse_transform(tf.transform(data)))
    original = extract_value_columns(data)
    for name in recovered:
        finite = np.isfinite(original[name])
        np.testing.assert_allclose(recovered[name][finite], original[name][finite], atol=1e-8)


@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
@pytest.mark.parametrize(
    "make_transformer",
    [
        lambda: DifferenceTransformer(periods=2),
        lambda: DetrendTransformer(trend="ct"),
        lambda: HamiltonFilterTransformer(h=2, p=2),
        lambda: HPFilterDetrend(lamb=1600.0),
    ],
    ids=["difference", "detrend", "hamilton", "hp"],
)
def test_batch_matches_per_series(make_transformer, method, positive_random_walk, batch_dataarray):
    tf = make_transformer().fit(positive_random_walk)
    assert_batch_matches_per_series(tf, batch_dataarray, method)
