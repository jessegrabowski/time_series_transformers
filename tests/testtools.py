import narwhals as nw
import numpy as np


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
