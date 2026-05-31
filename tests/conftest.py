import numpy as np
import pandas as pd
import polars as pl
import pytest

N = 50


@pytest.fixture(params=["pandas", "polars"])
def backend(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng()


def _dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _dated(backend, data):
    n = len(next(iter(data.values())))
    if backend == "pandas":
        return pd.DataFrame(data, index=pd.Index(_dates(n), name="date"))
    return pl.DataFrame({"date": _dates(n), **data})


@pytest.fixture
def make_dated(backend):
    """Build a constructor that wraps a column dict into the current backend.

    The frame carries a daily datetime axis — a ``DatetimeIndex`` for pandas,
    a ``date`` column for polars.
    """
    return lambda data: _dated(backend, data)


@pytest.fixture
def dated_random_walk(make_dated, rng):
    return make_dated(
        {"a": np.cumsum(rng.standard_normal(N)), "b": np.cumsum(rng.standard_normal(N))}
    )


@pytest.fixture
def dated_positive_random_walk(make_dated, rng):
    return make_dated(
        {
            "a": np.abs(np.cumsum(rng.standard_normal(N))) + 1.0,
            "b": np.abs(np.cumsum(rng.standard_normal(N))) + 1.0,
        }
    )


@pytest.fixture
def random_walk(rng):
    return pd.DataFrame(
        {"a": np.cumsum(rng.standard_normal(N)), "b": np.cumsum(rng.standard_normal(N))},
        index=pd.Index(_dates(N), name="date"),
    )


@pytest.fixture
def positive_random_walk(random_walk):
    return random_walk.abs() + 1.0
