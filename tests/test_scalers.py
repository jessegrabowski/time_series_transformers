import numpy as np
import pandas as pd
import pytest

from time_series_transformers import LogTransformer, PandasMinMaxScaler, PandasStandardScaler


@pytest.mark.parametrize("lam", [0, 0.25, 0.5, 1.0, 2.0])
def test_log_transformer_roundtrip(positive_random_walk, lam):
    tf = LogTransformer(lam=lam)
    result = tf.inverse_transform(tf.transform(positive_random_walk))
    pd.testing.assert_frame_equal(result, positive_random_walk, atol=1e-10)


def test_log_transformer_known_values():
    data = pd.DataFrame({"x": [1.0, np.e, np.e**2]})
    result = LogTransformer(lam=0).transform(data)
    np.testing.assert_allclose(result["x"].values, [0.0, 1.0, 2.0])


def test_log_transformer_lam1_is_shifted_identity(positive_random_walk):
    result = LogTransformer(lam=1.0).transform(positive_random_walk)
    pd.testing.assert_frame_equal(result, positive_random_walk - 1.0, atol=1e-12)


def test_standard_scaler_roundtrip(random_walk):
    tf = PandasStandardScaler().fit(random_walk)
    result = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(result, random_walk, atol=1e-12)


def test_standard_scaler_produces_zero_mean_unit_variance(random_walk):
    result = PandasStandardScaler().fit(random_walk).transform(random_walk)
    np.testing.assert_allclose(result.mean().values, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.std().values, 1.0, atol=1e-12)


def test_standard_scaler_zero_variance_column_passes_through():
    data = pd.DataFrame({"const": [5.0] * 10, "vary": range(10)})
    result = PandasStandardScaler().fit(data).transform(data)
    assert (result["const"] == 0.0).all()


def test_minmax_scaler_roundtrip(random_walk):
    tf = PandasMinMaxScaler().fit(random_walk)
    result = tf.inverse_transform(tf.transform(random_walk))
    pd.testing.assert_frame_equal(result, random_walk, atol=1e-12)


def test_minmax_scaler_produces_01_range(random_walk):
    result = PandasMinMaxScaler().fit(random_walk).transform(random_walk)
    np.testing.assert_allclose(result.min().values, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.max().values, 1.0, atol=1e-12)


def test_minmax_scaler_constant_column_passes_through():
    data = pd.DataFrame({"const": [3.0] * 10, "vary": range(10)})
    result = PandasMinMaxScaler().fit(data).transform(data)
    assert (result["const"] == 0.0).all()
