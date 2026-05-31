import numpy as np
import pandas as pd

from scipy.linalg import lstsq, solveh_banded
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from time_series_transformers._utils import to_dataframe

_TREND_COMPONENTS: dict[str, tuple[bool, bool, bool]] = {
    "c": (True, False, False),
    "t": (False, True, False),
    "ct": (True, True, False),
    "ctt": (True, True, True),
}


class DifferenceTransformer(BaseEstimator, TransformerMixin):
    r"""Periodic-difference transformer with cumulative-sum inverse.

    Computes :math:`x_t - x_{t - d}` where :math:`d` is ``periods``. Stores the
    first ``periods`` rows during :meth:`fit` so that :meth:`inverse_transform`
    can recover levels by cumulatively summing within each stride
    :math:`t \\bmod d`.

    Parameters
    ----------
    periods : int, optional
        Lookback length of the difference. Use ``periods=1`` for ordinary
        first differences and, e.g., ``periods=4`` for annual differences of
        quarterly data. Default 1.
    """

    def __init__(self, periods: int = 1) -> None:
        self.periods = periods

    def fit(self, X, y=None):
        if self.periods < 1:
            raise ValueError(f"periods must be >= 1, got {self.periods}.")
        X = to_dataframe(X)
        if X.shape[0] < self.periods:
            raise ValueError(
                f"Need at least periods={self.periods} observations to fit, got {X.shape[0]}."
            )
        self.initial_values_ = X.iloc[: self.periods].copy()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return to_dataframe(X).diff(periods=self.periods)

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        out = to_dataframe(X)
        p = self.periods
        n_seed = min(p, out.shape[0])
        for i in range(n_seed):
            out.iloc[i] = out.iloc[i].fillna(self.initial_values_.iloc[i])
        groups = np.arange(out.shape[0]) % p
        return out.groupby(groups).cumsum()


class DetrendTransformer(BaseEstimator, TransformerMixin):
    """Remove a deterministic trend via OLS.

    Parameters
    ----------
    trend : ``{"c", "t", "ct", "ctt"}``, default ``"c"``
        ``"c"`` — constant only,
        ``"t"`` — linear trend,
        ``"ct"`` — constant + linear,
        ``"ctt"`` — constant + linear + quadratic.
    """

    def __init__(self, trend: str = "c") -> None:
        self.trend = trend

    def fit(self, X, y=None):
        X = to_dataframe(X)
        F = self._feature_matrix(X.shape[0])

        betas: list[np.ndarray] = []
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            mask = np.isfinite(x)
            beta = lstsq(F[mask], x[mask])[0]
            betas.append(beta)

        self.params_ = np.column_stack(betas)
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        out = to_dataframe(X)
        trend = self._feature_matrix(out.shape[0]) @ self.params_
        out.loc[:, :] = out.to_numpy(dtype=float) - trend
        return out

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        out = to_dataframe(X)
        trend = self._feature_matrix(out.shape[0]) @ self.params_
        out.loc[:, :] = out.to_numpy(dtype=float) + trend
        return out

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


class HamiltonFilterTransformer(BaseEstimator, TransformerMixin):
    """Hamilton (2018) regression filter for trend–cycle decomposition.

    Decomposes each column into a *trend* (fitted values of a regression of
    ``y_{t+h}`` on ``y_t, y_{t-1}, …, y_{t-p+1}`` plus a constant) and a
    *cycle* (the residual).

    Parameters
    ----------
    h : int, default 2
        Forecasting horizon.
    p : int, default 1
        Number of lags in the autoregression.
    store_trend : bool, default True
        Whether to store the fitted trend.  Required for
        :meth:`inverse_transform`.

    Notes
    -----
    :meth:`inverse_transform` can only recover levels when ``store_trend`` is
    ``True`` **and** the index matches the data used in :meth:`fit`.
    """

    def __init__(self, h: int = 2, p: int = 1, store_trend: bool = True) -> None:
        self.h = h
        self.p = p
        self.store_trend = store_trend

    def fit(self, X, y=None):
        X = to_dataframe(X)
        self.columns_ = X.columns
        self.models_: dict[str, np.ndarray] = {}
        self.trends_: dict[str, pd.Series] = {}

        for col in X.columns:
            s = X[col].astype(float)
            _cycle, trend, beta = self._fit_one(s)
            self.models_[col] = beta
            if self.store_trend:
                self.trends_[col] = trend

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = to_dataframe(X)
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            out[col] = self._apply(X[col].astype(float), self.models_[col])
        return out

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        if not self.store_trend:
            raise ValueError("inverse_transform requires store_trend=True.")
        X = to_dataframe(X)
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            trend = self.trends_[col].reindex(X.index)
            out[col] = X[col].astype(float) + trend
        return out

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

    def _fit_one(self, s: pd.Series) -> tuple[pd.Series, pd.Series, np.ndarray]:
        vals = s.to_numpy(dtype=float)
        mask = np.isfinite(vals)
        idx = np.where(mask)[0]
        clean = vals[mask]

        t_idx, y, X = self._lagged_design(clean)
        beta = lstsq(X, y)[0]

        trend = np.full(len(clean), np.nan)
        trend[t_idx + self.h] = X @ beta
        cycle = clean - trend

        trend_s = pd.Series(np.nan, index=s.index, dtype=float)
        cycle_s = pd.Series(np.nan, index=s.index, dtype=float)
        trend_s.iloc[idx] = trend
        cycle_s.iloc[idx] = cycle

        return cycle_s, trend_s, beta

    def _apply(self, s: pd.Series, beta: np.ndarray) -> pd.Series:
        """Compute the cycle component of *s* using pre-fitted *beta*."""
        vals = s.to_numpy(dtype=float)
        mask = np.isfinite(vals)
        idx = np.where(mask)[0]
        clean = vals[mask]

        t_idx, _, X = self._lagged_design(clean)

        trend = np.full(len(clean), np.nan)
        trend[t_idx + self.h] = X @ beta
        cycle = clean - trend

        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.iloc[idx] = cycle
        return out


class HPFilterDetrend(BaseEstimator, TransformerMixin):
    r"""Hodrick–Prescott filter for trend–cycle decomposition.

    Decomposes each column into a smooth *trend* and a *cycle* (the residual)
    by solving the penalized least-squares problem

    .. math::

        \min_{\tau} \sum_t (y_t - \tau_t)^2
        + \lambda \sum_t \big[(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})\big]^2,

    where :math:`\lambda` penalizes curvature of the trend. :meth:`transform`
    returns the cycle :math:`y_t - \tau_t`; :meth:`inverse_transform` adds the
    trend stored during :meth:`fit` back to recover levels.

    Parameters
    ----------
    lamb : float, optional
        Smoothing parameter :math:`\lambda`. Larger values yield a smoother
        trend. Common choices are 1600 for quarterly data, 129600 for monthly
        data, and 6.25 for annual data. Default 1600.
    store_trend : bool, optional
        Whether to store the fitted trend. Required for
        :meth:`inverse_transform`. Default True.

    Notes
    -----
    :meth:`inverse_transform` can only recover levels when ``store_trend`` is
    ``True`` and the index matches the data used in :meth:`fit`.
    """

    def __init__(self, lamb: float = 1600.0, store_trend: bool = True) -> None:
        self.lamb = lamb
        self.store_trend = store_trend

    def fit(self, X, y=None):
        if self.lamb <= 0:
            raise ValueError(f"lamb must be positive, got {self.lamb}.")
        X = to_dataframe(X)
        self.columns_ = X.columns
        self.trends_: dict[str, pd.Series] = {}
        for col in X.columns:
            trend = self._trend(X[col].astype(float))
            if self.store_trend:
                self.trends_[col] = trend
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = to_dataframe(X)
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            s = X[col].astype(float)
            out[col] = s - self._trend(s)
        return out

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        if not self.store_trend:
            raise ValueError("inverse_transform requires store_trend=True.")
        X = to_dataframe(X)
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            trend = self.trends_[col].reindex(X.index)
            out[col] = X[col].astype(float) + trend
        return out

    def _trend(self, s: pd.Series) -> pd.Series:
        """Solve the HP problem for the trend of *s*, leaving NaN positions untouched."""
        vals = s.to_numpy(dtype=float)
        mask = np.isfinite(vals)
        idx = np.where(mask)[0]
        clean = vals[mask]

        n = len(clean)
        if n < 3:
            raise ValueError(f"Series too short ({n} obs); HP filter needs >= 3.")
        trend = solveh_banded(self._banded_lhs(n), clean)

        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.iloc[idx] = trend
        return out

    def _banded_lhs(self, n: int) -> np.ndarray:
        """Upper-banded storage of ``I + lamb * D2.T @ D2`` for :func:`solveh_banded`.

        ``D2`` is the ``(n - 2, n)`` second-difference operator, so the system
        matrix is symmetric, positive definite, and pentadiagonal; only the main
        diagonal and the two superdiagonals are built.
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
