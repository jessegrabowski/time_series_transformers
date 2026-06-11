import narwhals as nw
import numpy as np
import pandas as pd


def extract_value_columns(native):
    """Return the non-datetime columns of any frame as a dict of float arrays."""
    df = nw.from_native(native, eager_only=True)
    return {
        name: df.select(name).to_numpy()[:, 0].astype(float)
        for name, dtype in df.schema.items()
        if not dtype.is_temporal()
    }


def assert_columns_close(result, expected, **kwargs):
    """Assert the value columns of *result* match *expected* (a frame or name→array dict)."""
    got = extract_value_columns(result)
    want = expected if isinstance(expected, dict) else extract_value_columns(expected)
    assert got.keys() == want.keys()
    for name in got:
        np.testing.assert_allclose(got[name], want[name], **kwargs)


def assert_batch_matches_per_series(tf, da, method="transform", atol=1e-10):
    """Assert ``tf.<method>(da)`` matches the per-series frame transform, slice by slice.

    *da* must have dims ``(series, date, feature)`` with ``date`` and ``feature`` coordinates.
    Rebuild a dated frame from each series slice, run the same method on it, and compare values;
    also check the output keeps the input's dims and coordinates.
    """
    columns = list(da.coords["feature"].to_numpy())
    index = pd.Index(da.coords["date"].to_numpy(), name="date")
    batch = getattr(tf, method)(da, feature_dim="feature")

    assert batch.dims == da.dims
    assert batch.coords["date"].equals(da.coords["date"])
    assert batch.coords["feature"].equals(da.coords["feature"])
    for s in range(da.sizes["series"]):
        values = da.isel(series=s).transpose("date", "feature").to_numpy()
        ref = getattr(tf, method)(pd.DataFrame(values, index=index, columns=columns))
        got = batch.isel(series=s).transpose("date", "feature").to_numpy()
        want = np.column_stack([extract_value_columns(ref)[c] for c in columns])
        np.testing.assert_allclose(got, want, atol=atol, equal_nan=True)
