import numpy as np
import pandas as pd
import pytest

from time_series_transformers import (
    DetrendTransformer,
    DifferenceTransformer,
    HamiltonFilterTransformer,
    HPFilterDetrend,
)


def test_difference_roundtrip(random_walk):
    tf = DifferenceTransformer().fit(random_walk)
    result = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(result, random_walk, atol=1e-12)


def test_difference_known_values():
    data = pd.DataFrame({"x": [10.0, 13.0, 11.0]})
    result = DifferenceTransformer().fit(data).transform(data)
    np.testing.assert_allclose(result["x"].values[1:], [3.0, -2.0])


@pytest.mark.parametrize("periods", [1, 2, 4, 12])
def test_difference_periods_roundtrip(random_walk, periods):
    tf = DifferenceTransformer(periods=periods).fit(random_walk)
    result = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(result, random_walk, atol=1e-10)


def test_difference_periods_known_values():
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]})
    result = DifferenceTransformer(periods=2).fit(data).transform(data)
    assert result["x"].iloc[:2].isna().all()
    np.testing.assert_allclose(result["x"].values[2:], [3.0, 5.0, 7.0, 9.0])


def test_difference_periods_seasonal_known_values():
    data = pd.DataFrame({"x": np.arange(8, dtype=float) + 10.0})
    tf = DifferenceTransformer(periods=4).fit(data)
    diffed = tf.transform(data)
    assert diffed["x"].iloc[:4].isna().all()
    np.testing.assert_allclose(diffed["x"].values[4:], 4.0)
    recovered = tf.inverse_transform(diffed)
    pd.testing.assert_frame_equal(recovered, data, atol=1e-12)


def test_difference_invalid_periods_raises(random_walk):
    with pytest.raises(ValueError, match="periods must be >= 1"):
        DifferenceTransformer(periods=0).fit(random_walk)


def test_difference_short_series_raises():
    short = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least periods=4"):
        DifferenceTransformer(periods=4).fit(short)


@pytest.mark.parametrize("trend", ["c", "t", "ct", "ctt"])
def test_detrend_roundtrip(random_walk, trend):
    tf = DetrendTransformer(trend=trend).fit(random_walk)
    result = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(result, random_walk, atol=1e-10)


def test_detrend_constant_removes_mean(random_walk):
    result = DetrendTransformer(trend="c").fit(random_walk).transform(random_walk)
    np.testing.assert_allclose(result.mean().values, 0.0, atol=1e-10)


def test_detrend_linear_removes_linear_trend(rng):
    t = np.arange(100, dtype=float)
    data = pd.DataFrame({"x": 3.0 + 2.0 * t + rng.standard_normal(100) * 0.01})
    residuals = DetrendTransformer(trend="ct").fit(data).transform(data)
    assert residuals["x"].abs().max() < 0.1


def test_detrend_invalid_trend_raises(random_walk):
    with pytest.raises(ValueError, match="Invalid trend"):
        DetrendTransformer(trend="xyz").fit(random_walk)


def test_detrend_nan_values_do_not_corrupt_finite_values(random_walk):
    data = random_walk.copy()
    data.iloc[5, 0] = np.nan
    tf = DetrendTransformer(trend="ct").fit(data)
    recovered = tf.inverse_transform(tf.transform(data))
    finite = data.notna()
    np.testing.assert_allclose(
        recovered.values[finite.values], data.values[finite.values], atol=1e-10
    )


def test_detrend_transform_length_independent_of_fit(rng):
    train = pd.DataFrame({"x": rng.standard_normal(100)})
    test = pd.DataFrame({"x": rng.standard_normal(30)})
    tf = DetrendTransformer(trend="ct").fit(train)
    recovered = tf.inverse_transform(tf.transform(test))
    pd.testing.assert_frame_equal(recovered, test, atol=1e-10)


@pytest.mark.parametrize("h,p", [(1, 1), (2, 2), (4, 2), (8, 4)])
def test_hamilton_roundtrip(random_walk, h, p):
    tf = HamiltonFilterTransformer(h=h, p=p).fit(random_walk)
    cycle = tf.transform(random_walk)
    recovered = tf.inverse_transform(cycle)
    both_finite = recovered.notna() & random_walk.notna()
    np.testing.assert_allclose(
        recovered.values[both_finite.values],
        random_walk.values[both_finite.values],
        atol=1e-10,
    )


def test_hamilton_nan_structure(random_walk):
    h, p = 3, 2
    cycle = HamiltonFilterTransformer(h=h, p=p).fit(random_walk).transform(random_walk)
    n_leading_nan = h + p - 1
    assert cycle.iloc[:n_leading_nan].isna().all().all()
    assert cycle.iloc[n_leading_nan:].notna().all().all()


def test_hamilton_inverse_without_store_trend_raises(random_walk):
    tf = HamiltonFilterTransformer(store_trend=False).fit(random_walk)
    with pytest.raises(ValueError, match="store_trend"):
        tf.inverse_transform(tf.transform(random_walk))


def test_hamilton_short_series_raises():
    short = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="too short"):
        HamiltonFilterTransformer(h=2, p=2).fit(short)


def test_hp_filter_roundtrip(random_walk):
    tf = HPFilterDetrend(lamb=1600).fit(random_walk)
    recovered = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(recovered, random_walk, atol=1e-8)


@pytest.mark.parametrize("n", [3, 4, 5, 50])
def test_hp_filter_matches_dense_solve(rng, n):
    lamb = 1600.0
    y = rng.standard_normal(n)
    data = pd.DataFrame({"x": y})
    cycle = HPFilterDetrend(lamb=lamb).fit(data).transform(data)
    trend = y - cycle["x"].to_numpy()

    second_diff = np.diff(np.eye(n), n=2, axis=0)
    dense = np.eye(n) + lamb * (second_diff.T @ second_diff)
    np.testing.assert_allclose(trend, np.linalg.solve(dense, y), atol=1e-8)


def test_hp_filter_linear_series_has_zero_cycle():
    t = np.arange(100, dtype=float)
    data = pd.DataFrame({"x": 3.0 + 2.0 * t})
    cycle = HPFilterDetrend(lamb=1600).fit(data).transform(data)
    assert cycle["x"].abs().max() < 1e-6


def test_hp_filter_larger_lamb_yields_smoother_trend(random_walk):
    data = random_walk[["a"]]
    cycle_rough = HPFilterDetrend(lamb=100).fit(data).transform(data)
    cycle_smooth = HPFilterDetrend(lamb=1e7).fit(data).transform(data)
    trend_rough = (data - cycle_rough)["a"].to_numpy()
    trend_smooth = (data - cycle_smooth)["a"].to_numpy()
    assert np.abs(np.diff(trend_smooth, 2)).sum() < np.abs(np.diff(trend_rough, 2)).sum()


def test_hp_filter_inverse_without_store_trend_raises(random_walk):
    tf = HPFilterDetrend(store_trend=False).fit(random_walk)
    with pytest.raises(ValueError, match="store_trend"):
        tf.inverse_transform(tf.transform(random_walk))


@pytest.mark.parametrize("lamb", [-1, 0])
def test_hp_filter_invalid_lamb_raises(random_walk, lamb):
    with pytest.raises(ValueError, match="lamb must be positive"):
        HPFilterDetrend(lamb=lamb).fit(random_walk)


def test_hp_filter_short_series_raises():
    short = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="too short"):
        HPFilterDetrend().fit(short)


def test_hp_filter_nan_values_do_not_corrupt_finite_values(random_walk):
    data = random_walk.copy()
    data.iloc[5, 0] = np.nan
    tf = HPFilterDetrend().fit(data)
    recovered = tf.inverse_transform(tf.transform(data))
    finite = data.notna()
    np.testing.assert_allclose(
        recovered.values[finite.values], data.values[finite.values], atol=1e-8
    )
