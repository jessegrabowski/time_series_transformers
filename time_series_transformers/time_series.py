import numpy as np

from scipy.linalg import lstsq, solveh_banded

from time_series_transformers.base import TimeSeriesTransformer
from time_series_transformers.date_axis import align_by_date

_TREND_COMPONENTS: dict[str, tuple[bool, bool, bool]] = {
    "c": (True, False, False),
    "t": (False, True, False),
    "ct": (True, True, False),
    "ctt": (True, True, True),
}


class DifferenceTransformer(TimeSeriesTransformer):
    r"""Periodic difference :math:`x_t - x_{t-d}` (``d = periods``), inverted by cumsum.

    Parameters
    ----------
    periods : int, optional
        Difference lag — 1 for first differences, 4 for annual differences of quarterly
        data. Default 1.
    date_column : str, optional
        Time axis, auto-detected when ``None``. Default None.
    """

    def __init__(self, periods: int = 1, date_column: str | None = None) -> None:
        self.periods = periods
        self.date_column = date_column

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        if self.periods < 1:
            raise ValueError(f"periods must be >= 1, got {self.periods}.")
        if values.shape[0] < self.periods:
            raise ValueError(
                f"Need at least periods={self.periods} observations to fit, got {values.shape[0]}."
            )
        self.initial_values_ = values[: self.periods].copy()

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        p = self.periods
        out = np.full_like(values, np.nan)
        out[..., p:, :] = values[..., p:, :] - values[..., :-p, :]
        return out

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        p = self.periods
        out = values.copy()
        k = min(p, out.shape[-2])
        block = out[..., :k, :]
        nan_mask = np.isnan(block)
        fill = np.broadcast_to(self.initial_values_[:k], block.shape)
        block[nan_mask] = fill[nan_mask]
        for stride in range(p):
            out[..., stride::p, :] = np.cumsum(out[..., stride::p, :], axis=-2)
        return out


class DetrendTransformer(TimeSeriesTransformer):
    """Remove a deterministic trend via OLS.

    Parameters
    ----------
    trend : {"c", "t", "ct", "ctt"}, optional
        ``"c"`` constant, ``"t"`` linear, ``"ct"`` constant + linear, ``"ctt"`` adds
        quadratic. Default ``"c"``.
    date_column : str, optional
        Time axis, auto-detected when ``None``. Default None.
    """

    def __init__(self, trend: str = "c", date_column: str | None = None) -> None:
        self.trend = trend
        self.date_column = date_column

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        design = self._feature_matrix(values.shape[0])
        betas = []
        for column in range(values.shape[1]):
            x = values[:, column]
            mask = np.isfinite(x)
            betas.append(lstsq(design[mask], x[mask])[0])  # pyright: ignore[reportOptionalSubscript]
        self.params_ = np.column_stack(betas)

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return values - self._feature_matrix(values.shape[-2]) @ self.params_

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return values + self._feature_matrix(values.shape[-2]) @ self.params_

    def _feature_matrix(self, n_obs: int) -> np.ndarray:
        """Build the ``(n_obs, k)`` OLS design matrix for the chosen trend."""
        try:
            has_const, has_linear, has_quad = _TREND_COMPONENTS[self.trend]
        except KeyError:
            raise ValueError(
                f"Invalid trend {self.trend!r}. Choose from {list(_TREND_COMPONENTS)}."
            ) from None

        t = np.arange(n_obs, dtype=float)
        parts: list[np.ndarray] = []
        if has_const:
            parts.append(np.ones(n_obs))
        if has_linear:
            parts.append(t)
        if has_quad:
            parts.append(t**2)
        return np.column_stack(parts)


class HamiltonFilterTransformer(TimeSeriesTransformer):
    r"""Hamilton (2018) regression filter for trend–cycle decomposition.

    The trend is the fit of :math:`y_{t+h}` on :math:`y_t, \ldots, y_{t-p+1}` and a
    constant; the cycle is the residual.

    Parameters
    ----------
    h : int, optional
        Forecast horizon. Default 2.
    p : int, optional
        Autoregression lags. Default 1.
    store_trend : bool, optional
        Store the fitted trend; required for :meth:`inverse_transform`. Default True.
    date_column : str, optional
        Time axis, auto-detected when ``None``. Default None.
    """

    def __init__(
        self, h: int = 2, p: int = 1, store_trend: bool = True, date_column: str | None = None
    ) -> None:
        self.h = h
        self.p = p
        self.store_trend = store_trend
        self.date_column = date_column

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        self.betas_ = []
        trends = []
        for column in range(values.shape[1]):
            trend, beta = self._fit_one(values[:, column])
            self.betas_.append(beta)
            trends.append(trend)
        if self.store_trend:
            self.trend_values_ = np.column_stack(trends)
            self.trend_dates_ = dates

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                self._apply(values[:, column], self.betas_[column])
                for column in range(values.shape[1])
            ]
        )

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        if not self.store_trend:
            raise ValueError("inverse_transform requires store_trend=True.")
        return values + align_by_date(self.trend_dates_, self.trend_values_, dates)

    def _batch_apply(self, da, feature_dim, values_hook):
        # Per-column NaN masking and least-squares fits don't broadcast; loop per series.
        return self._batch_via_ufunc(da, feature_dim, values_hook, vectorize=True)

    def _lagged_design(self, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(t_idx, y, X)`` for the Hamilton autoregression."""
        n = len(vals)
        h, p = self.h, self.p
        if n <= h + p - 1:
            raise ValueError(f"Series too short ({n} obs) for h={h}, p={p}.")
        t_idx = np.arange(p - 1, n - h)
        y = vals[t_idx + h]
        X = np.column_stack([np.ones(len(t_idx))] + [vals[t_idx - j] for j in range(p)])
        return t_idx, y, X

    def _fit_one(self, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit the regression on the finite entries of *vals*, returning ``(trend, beta)``."""
        mask = np.isfinite(vals)
        idx = np.where(mask)[0]
        clean = vals[mask]

        t_idx, y, X = self._lagged_design(clean)
        beta = lstsq(X, y)[0]  # pyright: ignore[reportOptionalSubscript]

        trend = np.full(len(clean), np.nan)
        trend[t_idx + self.h] = X @ beta

        out = np.full(len(vals), np.nan)
        out[idx] = trend
        return out, beta

    def _apply(self, vals: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute the cycle component of *vals* using pre-fitted *beta*."""
        mask = np.isfinite(vals)
        idx = np.where(mask)[0]
        clean = vals[mask]

        t_idx, _, X = self._lagged_design(clean)
        trend = np.full(len(clean), np.nan)
        trend[t_idx + self.h] = X @ beta

        out = np.full(len(vals), np.nan)
        out[idx] = clean - trend
        return out


class HPFilterDetrend(TimeSeriesTransformer):
    r"""Hodrick–Prescott filter: split each column into a smooth trend and a cycle.

    Minimizes the penalized least squares

    .. math::

        \min_{\tau} \sum_t (y_t - \tau_t)^2
        + \lambda \sum_t \big[(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})\big]^2.

    Parameters
    ----------
    lamb : float, optional
        Smoothing :math:`\lambda` (larger = smoother). Typical: 1600 quarterly, 129600
        monthly, 6.25 annual. Default 1600.
    store_trend : bool, optional
        Store the fitted trend; required for :meth:`inverse_transform`. Default True.
    date_column : str, optional
        Time axis, auto-detected when ``None``. Default None.
    """

    def __init__(
        self, lamb: float = 1600.0, store_trend: bool = True, date_column: str | None = None
    ) -> None:
        self.lamb = lamb
        self.store_trend = store_trend
        self.date_column = date_column

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        if self.lamb <= 0:
            raise ValueError(f"lamb must be positive, got {self.lamb}.")
        if self.store_trend:
            trends = [self._trend(values[:, column]) for column in range(values.shape[1])]
            self.trend_values_ = np.column_stack(trends)
            self.trend_dates_ = dates

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                values[:, column] - self._trend(values[:, column])
                for column in range(values.shape[1])
            ]
        )

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        if not self.store_trend:
            raise ValueError("inverse_transform requires store_trend=True.")
        return values + align_by_date(self.trend_dates_, self.trend_values_, dates)

    def _batch_apply(self, da, feature_dim, values_hook):
        # Per-column banded solves don't broadcast; loop per series.
        return self._batch_via_ufunc(da, feature_dim, values_hook, vectorize=True)

    def _trend(self, vals: np.ndarray) -> np.ndarray:
        """Solve the HP problem for the trend of *vals*, leaving NaN positions untouched."""
        mask = np.isfinite(vals)
        idx = np.where(mask)[0]
        clean = vals[mask]

        n = len(clean)
        if n < 3:
            raise ValueError(f"Series too short ({n} obs); HP filter needs >= 3.")
        trend = solveh_banded(self._banded_lhs(n), clean)

        out = np.full(len(vals), np.nan)
        out[idx] = trend
        return out

    def _banded_lhs(self, n: int) -> np.ndarray:
        """Upper-banded storage of ``I + lamb * D2.T @ D2`` for :func:`solveh_banded`.

        ``D2`` is the ``(n - 2, n)`` second-difference operator, so the system matrix is
        symmetric, positive definite, and pentadiagonal; only the main diagonal and the
        two superdiagonals are built.
        """
        lamb = self.lamb
        i = np.arange(n)
        diag = (i <= n - 3) + 4.0 * ((i >= 1) & (i <= n - 2)) + (i >= 2)
        j = np.arange(n - 1)
        super1 = -2.0 * (j <= n - 3) - 2.0 * (j >= 1)

        ab = np.zeros((3, n))
        ab[0, 2:] = lamb
        ab[1, 1:] = lamb * super1
        ab[2, :] = 1.0 + lamb * diag
        return ab
