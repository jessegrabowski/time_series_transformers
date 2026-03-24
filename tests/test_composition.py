import pandas as pd
import pytest

from time_series_transformers import (
    DataFrameFeatureUnion,
    DifferenceTransformer,
    InvertiblePipeline,
    LogTransformer,
    PandasMinMaxScaler,
    PandasStandardScaler,
)


def test_pipeline_roundtrip(positive_random_walk):
    pipe = InvertiblePipeline(
        [
            ("log", LogTransformer(lam=0)),
            ("scale", PandasStandardScaler()),
        ]
    )
    pipe.fit(positive_random_walk)
    result = pipe.inverse_transform(pipe.transform(positive_random_walk))
    pd.testing.assert_frame_equal(result, positive_random_walk, atol=1e-10)


def test_pipeline_applies_steps_in_declared_order():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pipe = InvertiblePipeline(
        [
            ("diff", DifferenceTransformer()),
            ("scale", PandasStandardScaler()),
        ]
    )
    pipe.fit(data)
    result = pipe.transform(data)
    # diff produces NaN in row 0; if order were wrong this would not survive scaling
    assert result.iloc[0].isna().all()


def test_pipeline_non_invertible_step_raises(random_walk):
    class ForwardOnly:
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X

    pipe = InvertiblePipeline([("fwd", ForwardOnly())])
    pipe.fit(random_walk)
    with pytest.raises(TypeError, match="does not support inverse_transform"):
        pipe.inverse_transform(random_walk)


def test_feature_union_roundtrip(positive_random_walk):
    union = DataFrameFeatureUnion(
        [
            (
                "log_group",
                ["a"],
                InvertiblePipeline(
                    [
                        ("log", LogTransformer(lam=0)),
                        ("scale", PandasStandardScaler()),
                    ]
                ),
            ),
            ("scale_group", ["b"], PandasMinMaxScaler()),
        ]
    )
    union.fit(positive_random_walk)
    result = union.inverse_transform(union.transform(positive_random_walk))
    pd.testing.assert_frame_equal(result, positive_random_walk, atol=1e-10)


def test_feature_union_preserves_declaration_order(random_walk):
    union = DataFrameFeatureUnion(
        [
            ("second", ["b"], PandasStandardScaler()),
            ("first", ["a"], PandasMinMaxScaler()),
        ]
    )
    union.fit(random_walk)
    result = union.transform(random_walk)
    assert list(result.columns) == ["b", "a"]
