import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def random_walk(rng):
    return pd.DataFrame({
        "a": np.cumsum(rng.standard_normal(50)),
        "b": np.cumsum(rng.standard_normal(50)),
    })


@pytest.fixture
def positive_random_walk(random_walk):
    return random_walk.abs() + 1.0
