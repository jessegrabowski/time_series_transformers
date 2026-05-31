import numpy as np

from time_series_transformers.base import TimeSeriesTransformer


class LogTransformer(TimeSeriesTransformer):
    r"""Log (``lam=0``) or signed-power Box–Cox transform.

    .. math::

        \frac{\operatorname{sign}(x)\,|x|^{\lambda} - 1}{\lambda}.

    Parameters
    ----------
    lam : float, optional
        Power :math:`\lambda`; ``0`` is the log, ``1`` the shifted identity. Default 0.0.
    date_column : str, optional
        Time axis, auto-detected when ``None``. Default None.
    """

    def __init__(self, lam: float = 0.0, date_column: str | None = None) -> None:
        self.lam = lam
        self.date_column = date_column

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        if self.lam == 0:
            return np.log(values)
        return (np.sign(values) * np.abs(values) ** self.lam - 1.0) / self.lam

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        if self.lam == 0:
            return np.exp(values)
        inner = self.lam * values + 1.0
        return np.sign(inner) * np.abs(inner) ** (1.0 / self.lam)


class TimeSeriesStandardScaler(TimeSeriesTransformer):
    """Column-wise z-score normalization.

    Uses sample std (``ddof=1``); zero-variance columns pass through unchanged.
    """

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        self.mean_ = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        std[std == 0.0] = 1.0
        self.std_ = std

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return values * self.std_ + self.mean_


class TimeSeriesMinMaxScaler(TimeSeriesTransformer):
    """Column-wise min–max scaling to ``[0, 1]``; constant columns pass through unchanged."""

    def _fit_values(self, values: np.ndarray, dates: np.ndarray) -> None:
        self.min_ = np.nanmin(values, axis=0)
        self.max_ = np.nanmax(values, axis=0)

    def _transform_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        denom[denom == 0.0] = 1.0
        return (values - self.min_) / denom

    def _inverse_values(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        return (self.max_ - self.min_) * values + self.min_
