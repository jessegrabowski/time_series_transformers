import sys

from typing import Protocol

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from time_series_transformers.date_axis import replace_columns, restore_frame, to_dated_frame


def _is_dataarray(X) -> bool:
    """Report whether *X* is an ``xarray.DataArray`` without importing xarray."""
    # A DataArray can't exist unless xarray is already imported, so a missing module rules it
    # out and the dataframe path never pays an xarray import.
    xr = sys.modules.get("xarray")
    return xr is not None and isinstance(X, xr.DataArray)


def _require_xarray():
    try:
        import xarray as xr  # noqa: PLC0415  (optional dependency, imported lazily)
    except ImportError as exc:
        raise ImportError(
            "Batch transformation of xarray inputs requires the optional 'xarray' dependency. "
            "Install it with `pip install time_series_transformers[xarray]`."
        ) from exc
    return xr


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

    Once fit, :meth:`transform` and :meth:`inverse_transform` also accept an
    ``xarray.DataArray`` carrying extra batch dimensions (many series, regions, scenarios)
    alongside the time and feature axes. The fitted parameters broadcast across the batch
    dimensions; see :meth:`transform` for the required ``feature_dim`` argument and the
    coordinate-matching contract. Fitting on a DataArray is not supported.

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
        if _is_dataarray(X):
            raise TypeError(
                "Cannot fit on an xarray.DataArray. Fit on a dataframe, then pass batched "
                "DataArrays to transform / inverse_transform."
            )
        df, name, index_meta = to_dated_frame(X, self.date_column)
        self.date_column_ = name
        self.index_meta_ = index_meta
        self.columns_ = [column for column in df.columns if column != name]
        dates = df.get_column(name).to_numpy()
        self.dates_ = dates
        values = df.select(self.columns_).to_numpy().astype(float)
        self._fit_values(values, dates)
        return self

    def transform(self, X, y=None, *, feature_dim: str | None = None):
        """Transform *X*, a dataframe or an ``xarray.DataArray``.

        Parameters
        ----------
        X : dataframe or xarray.DataArray
            A narwhals-supported dataframe, or a DataArray whose time axis is a dimension
            named after the fitted date column. The DataArray may carry any number of extra
            batch dimensions; the fitted parameters broadcast across them.
        feature_dim : str, optional
            For a DataArray, the dimension holding the value columns. Required for DataArray
            inputs and rejected otherwise. Its length must match the fitted columns, and if
            it carries coordinate labels they must equal the fitted columns in order. Default
            None.
        """
        check_is_fitted(self)
        if _is_dataarray(X):
            return self._batch_apply(X, feature_dim, self._transform_values)
        if feature_dim is not None:
            raise ValueError("feature_dim only applies to xarray.DataArray inputs.")
        df, name, index_meta = to_dated_frame(X, self.date_column or self.date_column_)
        values = df.select(self.columns_).to_numpy().astype(float)
        out = self._transform_values(values, df.get_column(name).to_numpy())
        return restore_frame(replace_columns(df, self.columns_, out), name, index_meta)

    def inverse_transform(self, X, y=None, *, feature_dim: str | None = None):
        """Invert :meth:`transform`. Accepts the same dataframe or DataArray inputs."""
        check_is_fitted(self)
        if _is_dataarray(X):
            return self._batch_apply(X, feature_dim, self._inverse_values)
        if feature_dim is not None:
            raise ValueError("feature_dim only applies to xarray.DataArray inputs.")
        df, name, index_meta = to_dated_frame(X, self.date_column or self.date_column_)
        values = df.select(self.columns_).to_numpy().astype(float)
        out = self._inverse_values(values, df.get_column(name).to_numpy())
        return restore_frame(replace_columns(df, self.columns_, out), name, index_meta)

    def _batch_apply(self, da, feature_dim: str | None, values_hook):
        """Apply a value hook across a DataArray's batch dims.

        The default broadcasts the hook over the batch dims in a single call. Subclasses whose
        hooks work one series at a time (per-column linear solves) override this to loop.
        """
        return self._batch_via_ufunc(da, feature_dim, values_hook, vectorize=False)

    def _batch_via_ufunc(self, da, feature_dim: str | None, values_hook, *, vectorize: bool):
        """Validate, run the hook through :func:`xr.apply_ufunc`, and restore the dim order.

        With ``vectorize=False`` the hook sees the full ``(..., n_obs, n_columns)`` array and
        broadcasts over the batch dims; with ``vectorize=True`` the batch dims are looped and
        the hook sees one 2D slice at a time.
        """
        xr = _require_xarray()
        self._validate_batch(da, feature_dim)
        dates = self.dates_
        core_dims = [self.date_column_, feature_dim]

        def block_hook(block: np.ndarray) -> np.ndarray:
            return values_hook(block.astype(float), dates)

        result = xr.apply_ufunc(
            block_hook,
            da,
            input_core_dims=[core_dims],
            output_core_dims=[core_dims],
            vectorize=vectorize,
        )
        return result.transpose(*da.dims)

    def _validate_batch(self, da, feature_dim: str | None) -> None:
        """Validate a DataArray's time and feature axes against the fitted state."""
        if feature_dim is None:
            raise ValueError("feature_dim is required when transforming an xarray.DataArray.")

        time_dim = self.date_column_
        if time_dim not in da.dims:
            raise ValueError(
                f"DataArray has no {time_dim!r} dimension; the time axis must be named after the "
                f"fitted date column. Found dims {tuple(da.dims)}."
            )
        if time_dim not in da.coords:
            raise ValueError(
                f"DataArray dimension {time_dim!r} has no coordinate labels to validate against "
                "the fitted time axis."
            )
        if not np.array_equal(da.coords[time_dim].to_numpy(), self.dates_):
            raise ValueError(
                f"The {time_dim!r} coordinate does not match the time axis seen at fit time; batch "
                "transform requires identical dates in identical order."
            )

        if feature_dim not in da.dims:
            raise ValueError(
                f"DataArray has no {feature_dim!r} dimension. Found dims {tuple(da.dims)}."
            )
        if da.sizes[feature_dim] != len(self.columns_):
            raise ValueError(
                f"The {feature_dim!r} dimension has length {da.sizes[feature_dim]}, but the "
                f"transformer was fit on {len(self.columns_)} columns {self.columns_}."
            )
        if feature_dim in da.coords:
            labels = list(da.coords[feature_dim].to_numpy())
            if labels != list(self.columns_):
                raise ValueError(
                    f"The {feature_dim!r} coordinate {labels} does not match the columns seen at "
                    f"fit time {self.columns_}; batch transform requires identical features in "
                    "identical order."
                )
