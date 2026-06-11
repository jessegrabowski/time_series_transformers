import numpy as np
import pandas as pd
import pytest

from tests.testtools import extract_value_columns
from time_series_transformers import HPFilterDetrend, TimeSeriesStandardScaler

xr = pytest.importorskip("xarray")

COLUMNS = ["a", "b"]


@pytest.fixture
def scaler(positive_random_walk):
    return TimeSeriesStandardScaler().fit(positive_random_walk)


def test_batch_preserves_arbitrary_dim_order(scaler, dates, rng):
    """Time and feature need not be last; output keeps the input's dim order."""
    n = len(dates)
    data = np.abs(np.cumsum(rng.standard_normal((len(COLUMNS), 2, n)), axis=-1)) + 1.0
    da = xr.DataArray(
        data,
        dims=("feature", "region", "date"),
        coords={"date": dates, "feature": COLUMNS},
    )
    out = scaler.transform(da, feature_dim="feature")
    assert out.dims == da.dims
    index = pd.Index(dates, name="date")
    for r in range(da.sizes["region"]):
        values = da.isel(region=r).transpose("date", "feature").to_numpy()
        ref = scaler.transform(pd.DataFrame(values, index=index, columns=COLUMNS))
        got = out.isel(region=r).transpose("date", "feature").to_numpy()
        want = np.column_stack([extract_value_columns(ref)[c] for c in COLUMNS])
        np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("make_transformer", [TimeSeriesStandardScaler, HPFilterDetrend])
def test_batch_over_multiple_batch_dims(make_transformer, positive_random_walk, dates, rng):
    """Both the broadcast and per-series loop paths handle more than one batch dim."""
    n = len(dates)
    data = np.abs(np.cumsum(rng.standard_normal((2, 3, n, len(COLUMNS))), axis=2)) + 1.0
    da = xr.DataArray(
        data,
        dims=("series", "scenario", "date", "feature"),
        coords={"date": dates, "feature": COLUMNS},
    )
    tf = make_transformer().fit(positive_random_walk)
    out = tf.transform(da, feature_dim="feature")
    assert out.dims == da.dims
    index = pd.Index(dates, name="date")
    for s in range(da.sizes["series"]):
        for sc in range(da.sizes["scenario"]):
            values = da.isel(series=s, scenario=sc).transpose("date", "feature").to_numpy()
            ref = tf.transform(pd.DataFrame(values, index=index, columns=COLUMNS))
            got = out.isel(series=s, scenario=sc).transpose("date", "feature").to_numpy()
            want = np.column_stack([extract_value_columns(ref)[c] for c in COLUMNS])
            np.testing.assert_allclose(got, want, atol=1e-10)


def test_batch_without_extra_dims(scaler, dates, rng):
    """A plain (date, feature) DataArray transforms like the equivalent frame."""
    n = len(dates)
    data = np.abs(np.cumsum(rng.standard_normal((n, len(COLUMNS))), axis=0)) + 1.0
    da = xr.DataArray(data, dims=("date", "feature"), coords={"date": dates, "feature": COLUMNS})
    out = scaler.transform(da, feature_dim="feature")
    ref = scaler.transform(pd.DataFrame(data, index=pd.Index(dates, name="date"), columns=COLUMNS))
    want = np.column_stack([extract_value_columns(ref)[c] for c in COLUMNS])
    np.testing.assert_allclose(out.transpose("date", "feature").to_numpy(), want, atol=1e-10)


def test_feature_dim_without_coords_succeeds(scaler, dates, rng):
    """Feature labels are optional; a correctly-sized feature dim without coords transforms."""
    n = len(dates)
    data = np.abs(np.cumsum(rng.standard_normal((n, len(COLUMNS))), axis=0)) + 1.0
    da = xr.DataArray(data, dims=("date", "feature"), coords={"date": dates})
    out = scaler.transform(da, feature_dim="feature")
    ref = scaler.transform(pd.DataFrame(data, index=pd.Index(dates, name="date"), columns=COLUMNS))
    want = np.column_stack([extract_value_columns(ref)[c] for c in COLUMNS])
    np.testing.assert_allclose(out.transpose("date", "feature").to_numpy(), want, atol=1e-10)


def test_fit_on_dataarray_raises(batch_dataarray):
    with pytest.raises(TypeError, match="Cannot fit on an xarray"):
        TimeSeriesStandardScaler().fit(batch_dataarray)


def test_feature_dim_required_for_dataarray(scaler, batch_dataarray):
    with pytest.raises(ValueError, match="feature_dim is required"):
        scaler.transform(batch_dataarray)


def test_feature_dim_rejected_for_frame(scaler, positive_random_walk):
    with pytest.raises(ValueError, match="feature_dim only applies"):
        scaler.transform(positive_random_walk, feature_dim="feature")


def test_missing_time_dim_raises(scaler, rng):
    da = xr.DataArray(
        rng.standard_normal((3, len(COLUMNS))),
        dims=("series", "feature"),
        coords={"feature": COLUMNS},
    )
    with pytest.raises(ValueError, match="no 'date' dimension"):
        scaler.transform(da, feature_dim="feature")


def test_time_coord_mismatch_raises(scaler, batch_dataarray):
    shifted = batch_dataarray.assign_coords(
        date=batch_dataarray.coords["date"] + pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError, match="does not match the time axis"):
        scaler.transform(shifted, feature_dim="feature")


def test_missing_time_coord_raises(scaler, dates, rng):
    da = xr.DataArray(
        rng.standard_normal((len(dates), len(COLUMNS))),
        dims=("date", "feature"),
        coords={"feature": COLUMNS},
    )
    with pytest.raises(ValueError, match="no coordinate labels"):
        scaler.transform(da, feature_dim="feature")


def test_feature_length_mismatch_raises(scaler, dates, rng):
    da = xr.DataArray(
        rng.standard_normal((len(dates), 3)),
        dims=("date", "feature"),
        coords={"date": dates, "feature": ["a", "b", "c"]},
    )
    with pytest.raises(ValueError, match="was fit on 2 columns"):
        scaler.transform(da, feature_dim="feature")


def test_feature_coord_mismatch_raises(scaler, dates, rng):
    da = xr.DataArray(
        rng.standard_normal((len(dates), len(COLUMNS))),
        dims=("date", "feature"),
        coords={"date": dates, "feature": ["b", "a"]},
    )
    with pytest.raises(ValueError, match="does not match the columns"):
        scaler.transform(da, feature_dim="feature")
