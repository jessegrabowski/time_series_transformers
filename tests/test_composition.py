import numpy as np
import pytest

from sklearn.base import BaseEstimator

from tests.testtools import assert_columns_close, extract_value_columns
from time_series_transformers import (
    DataFrameFeatureUnion,
    DifferenceTransformer,
    InvertiblePipeline,
    LogTransformer,
    TimeSeriesMinMaxScaler,
    TimeSeriesStandardScaler,
)


def test_pipeline_roundtrip(dated_positive_random_walk):
    X = dated_positive_random_walk
    pipe = InvertiblePipeline(
        [("log", LogTransformer(lam=0)), ("scale", TimeSeriesStandardScaler())]
    ).fit(X)
    assert_columns_close(pipe.inverse_transform(pipe.transform(X)), X, atol=1e-10)


def test_pipeline_applies_steps_in_declared_order(make_dated):
    data = make_dated({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pipe = InvertiblePipeline(
        [("diff", DifferenceTransformer()), ("scale", TimeSeriesStandardScaler())]
    ).fit(data)
    result = pipe.transform(data)
    assert np.isnan(extract_value_columns(result)["x"][0])


def test_pipeline_non_invertible_step_raises(dated_random_walk):
    class ForwardOnly(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X

    pipe = InvertiblePipeline([("fwd", ForwardOnly())]).fit(dated_random_walk)
    with pytest.raises(TypeError, match="does not support inverse_transform"):
        pipe.inverse_transform(dated_random_walk)


def test_feature_union_roundtrip(dated_positive_random_walk):
    X = dated_positive_random_walk
    union = DataFrameFeatureUnion(
        [
            (
                "log_group",
                ["a"],
                InvertiblePipeline(
                    [("log", LogTransformer(lam=0)), ("scale", TimeSeriesStandardScaler())]
                ),
            ),
            ("scale_group", ["b"], TimeSeriesMinMaxScaler()),
        ]
    ).fit(X)
    assert_columns_close(union.inverse_transform(union.transform(X)), X, atol=1e-10)


def test_feature_union_preserves_declaration_order(dated_random_walk):
    union = DataFrameFeatureUnion(
        [("second", ["b"], TimeSeriesStandardScaler()), ("first", ["a"], TimeSeriesMinMaxScaler())]
    ).fit(dated_random_walk)
    result = union.transform(dated_random_walk)
    assert list(extract_value_columns(result).keys()) == ["b", "a"]


def test_feature_union_clones_shared_pipelines(dated_random_walk):
    shared = TimeSeriesStandardScaler()
    union = DataFrameFeatureUnion([("first", ["a"], shared), ("second", ["b"], shared)]).fit(
        dated_random_walk
    )
    result = extract_value_columns(union.transform(dated_random_walk))
    assert list(result.keys()) == ["a", "b"]
    assert all(np.isfinite(col).all() for col in result.values())
