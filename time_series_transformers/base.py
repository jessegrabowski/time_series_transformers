from typing import Protocol

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from time_series_transformers.date_axis import replace_columns, restore_frame, to_dated_frame


class Transformer(Protocol):
    """The fit/transform interface a pipeline or union step must provide."""

    def fit(self, X, y=None) -> "Transformer": ...

    def transform(self, X, y=None): ...


class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    r"""Backend-agnostic, date-aware base for the package's transformers.

    Built on narwhals, so it accepts and returns any supported dataframe. Resolves the
    single datetime axis, carries it through untouched, and hands the value columns to the
    subclass numpy hooks (:meth:`_fit_values`, :meth:`_transform_values`,
    :meth:`_inverse_values`) as an ``(n_obs, n_columns)`` array plus the ``(n_obs,)``
    date keys.

    Parameters
    ----------
    date_column : str, optional
        The datetime column or index marking the time axis. The frame must carry exactly
        one; a ``ValueError`` lists the offenders otherwise. Auto-detected when ``None``.
        Default None.
    """

    def __init__(self, date_column: str | None = None) -> None:
        self.date_column = date_column

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        """Learn state from the value array. Default: stateless."""

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X, y=None):
        df, name, index_meta = to_dated_frame(X, self.date_column)
        self.date_column_ = name
        self.index_meta_ = index_meta
        self.columns_ = [column for column in df.columns if column != name]
        values = df.select(self.columns_).to_numpy().astype(float)
        self._fit_values(values, df.get_column(name).to_numpy())
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        df, name, index_meta = to_dated_frame(X, self.date_column or self.date_column_)
        values = df.select(self.columns_).to_numpy().astype(float)
        out = self._transform_values(values, df.get_column(name).to_numpy())
        return restore_frame(replace_columns(df, self.columns_, out), name, index_meta)

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        df, name, index_meta = to_dated_frame(X, self.date_column or self.date_column_)
        values = df.select(self.columns_).to_numpy().astype(float)
        out = self._inverse_values(values, df.get_column(name).to_numpy())
        return restore_frame(replace_columns(df, self.columns_, out), name, index_meta)
