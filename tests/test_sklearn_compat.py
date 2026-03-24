import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from time_series_transformers import (
    DetrendTransformer,
    DifferenceTransformer,
    HamiltonFilterTransformer,
    LogTransformer,
    PandasMinMaxScaler,
    PandasStandardScaler,
)

ALL_TRANSFORMERS = [
    LogTransformer(),
    LogTransformer(lam=0.5),
    PandasStandardScaler(),
    PandasMinMaxScaler(),
    DifferenceTransformer(),
    DetrendTransformer(),
    DetrendTransformer(trend="ctt"),
    HamiltonFilterTransformer(),
    HamiltonFilterTransformer(h=4, p=3),
]

STATEFUL_TRANSFORMERS = [
    PandasStandardScaler(),
    PandasMinMaxScaler(),
    DifferenceTransformer(),
    DetrendTransformer(),
    HamiltonFilterTransformer(),
]


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS, ids=repr)
def test_sklearn_clone(transformer):
    clone(transformer)


@pytest.mark.parametrize("transformer", STATEFUL_TRANSFORMERS, ids=repr)
def test_unfitted_transform_raises(transformer, random_walk):
    with pytest.raises(NotFittedError):
        transformer.transform(random_walk)


def test_transform_does_not_mutate_input(random_walk):
    original = random_walk.copy()
    DetrendTransformer(trend="ct").fit(random_walk).transform(random_walk)
    pd.testing.assert_frame_equal(random_walk, original)

