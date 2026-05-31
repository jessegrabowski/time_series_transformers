from typing import Protocol, cast, runtime_checkable

import narwhals as nw

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from time_series_transformers.base import Transformer
from time_series_transformers.date_axis import restore_frame, to_dated_frame


@runtime_checkable
class _Invertible(Protocol):
    def inverse_transform(self, X, y=None): ...


class InvertiblePipeline(BaseEstimator, TransformerMixin):
    """Chain transformers; :meth:`inverse_transform` walks them in reverse.

    Parameters
    ----------
    steps : list of (name, transformer) tuples
        The transformers to apply in order.
    """

    def __init__(self, steps: list[tuple[str, Transformer]]) -> None:
        self.steps = steps

    def fit(self, X, y=None):
        Xt = X
        self.fitted_steps_: list[tuple[str, Transformer]] = []
        for name, outer_step in self.steps:
            step = cast(Transformer, clone(outer_step))
            Xt = step.fit(Xt, y).transform(Xt, y)
            self.fitted_steps_.append((name, step))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        Xt = X
        for _, step in self.fitted_steps_:
            Xt = step.transform(Xt, y)
        return Xt

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        Xt = X
        for name, step in reversed(self.fitted_steps_):
            if not isinstance(step, _Invertible):
                raise TypeError(
                    f"Step {name!r} ({type(step).__name__}) does not support inverse_transform."
                )
            Xt = step.inverse_transform(Xt, y)
        return Xt


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """Apply a transformer to each column group, then concatenate the results.

    Parameters
    ----------
    transformers : list of (name, columns, transformer) tuples
        ``columns`` is the value-column names handed to the transformer; the date axis is
        supplied automatically.
    date_column : str, optional
        Time axis shared across groups, auto-detected when ``None``. Default None.
    """

    def __init__(
        self,
        transformers: list[tuple[str, list[str], Transformer]],
        date_column: str | None = None,
    ) -> None:
        self.transformers = transformers
        self.date_column = date_column

    def fit(self, X, y=None):
        df, name, _ = to_dated_frame(X, self.date_column)
        self.date_column_ = name
        self.fitted_: list[tuple[str, list[str], Transformer]] = []
        for label, columns, transformer in self.transformers:
            group = nw.to_native(df.select(name, *columns))
            fitted = cast(Transformer, clone(transformer)).fit(group, y)
            self.fitted_.append((label, columns, fitted))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self._recombine(X, lambda transformer, group: transformer.transform(group, y))

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        return self._recombine(
            X, lambda transformer, group: transformer.inverse_transform(group, y)
        )

    def _recombine(self, X, apply):
        df, name, index_meta = to_dated_frame(X, self.date_column or self.date_column_)
        pieces = [df.select(name)]
        for _, columns, transformer in self.fitted_:
            group = nw.to_native(df.select(name, *columns))
            result = nw.from_native(apply(transformer, group), eager_only=True)
            pieces.append(result.select(columns))
        return restore_frame(nw.concat(pieces, how="horizontal"), name, index_meta)
