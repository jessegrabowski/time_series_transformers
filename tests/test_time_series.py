import numpy as np
import pandas as pd
import pytest

from time_series_transformers import (
    DetrendTransformer,
    DifferenceTransformer,
    HamiltonFilterTransformer,
)


def test_difference_roundtrip(random_walk):
    tf = DifferenceTransformer().fit(random_walk)
    result = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(result, random_walk, atol=1e-12)


def test_difference_values():
    data = pd.DataFrame({"x": [10.0, 13.0, 11.0]})
    result = DifferenceTransformer().fit(data).transform(data)
    np.testing.assert_allclose(result["x"].values[1:], [3.0, -2.0])


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

