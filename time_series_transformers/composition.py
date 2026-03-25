import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


class InvertiblePipeline(BaseEstimator, TransformerMixin):
    """Sequential pipeline whose :meth:`inverse_transform` walks steps in reverse.

    Parameters
    ----------
    steps : list of (name, transformer) tuples
    """

    def __init__(self, steps: list[tuple[str, BaseEstimator]]) -> None:
        self.steps = steps

    def fit(self, X, y=None):
        Xt = X.copy()
        self.fitted_steps_: list[tuple[str, BaseEstimator]] = []
        for name, outer_step in self.steps:
            step = clone(outer_step)
            Xt = step.fit(Xt, y).transform(Xt, y)
            self.fitted_steps_.append((name, step))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        Xt = X.copy()
        for _, step in self.fitted_steps_:
            Xt = step.transform(Xt, y)
        return Xt

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        Xt = X.copy()
        for name, step in reversed(self.fitted_steps_):
            if not hasattr(step, "inverse_transform"):
                raise TypeError(
                    f"Step {name!r} ({type(step).__name__}) does not support inverse_transform."
                )
            Xt = step.inverse_transform(Xt, y)
        return Xt


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """Apply different pipelines to different column groups, then concatenate.

    Parameters
    ----------
    transformers : list of (name, columns, transformer) tuples
        *columns* is a list of column names to pass to the transformer.
    """

    def __init__(self, transformers: list[tuple[str, list[str], BaseEstimator]]) -> None:
        self.transformers = transformers

    def fit(self, X, y=None):
        self.fitted_: list[tuple[str, list[str], BaseEstimator]] = []
        for name, cols, transformer in self.transformers:
            fitted = clone(transformer).fit(X[cols], y)
            self.fitted_.append((name, cols, fitted))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        pieces = []
        for _, cols, transformer in self.fitted_:
            Xt = transformer.transform(X[cols], y)
            Xt = Xt.copy()
            Xt.columns = cols
            pieces.append(Xt)
        return pd.concat(pieces, axis=1).reindex(columns=self._all_columns())

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        pieces = []
        for _, cols, transformer in self.fitted_:
            Xt = transformer.inverse_transform(X[cols], y)
            Xt = Xt.copy()
            Xt.columns = cols
            pieces.append(Xt)
        return pd.concat(pieces, axis=1).reindex(columns=self._all_columns())

    def _all_columns(self) -> list[str]:
        seen: list[str] = []
        for _, cols, _ in self.fitted_:
            seen.extend(cols)
        return seen
