import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from time_series_transformers._utils import to_dataframe


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log / signed-power (Box-Cox) transformer.

    When ``lam=0`` the transform is the natural logarithm.
    Otherwise, the signed generalized Box-Cox transform is applied::

        (sign(x) · |x|^lam − 1) / lam

    Parameters
    ----------
    lam : float, default 0.0
        Power parameter.  ``0`` → log, ``1`` → identity (shifted by −1).
    """

    def __init__(self, lam: float = 0.0) -> None:
        self.lam = lam

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = to_dataframe(X)
        if self.lam == 0:
            return np.log(X)
        return (np.sign(X) * np.abs(X) ** self.lam - 1.0) / self.lam

    def inverse_transform(self, X, y=None):
        X = to_dataframe(X)
        if self.lam == 0:
            return np.exp(X)
        inner = self.lam * X + 1.0
        return np.sign(inner) * np.abs(inner) ** (1.0 / self.lam)


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    """Column-wise z-score normalization preserving DataFrame structure.

    Uses the sample standard deviation (``ddof=1``, the pandas default).
    Zero-variance columns are left unscaled.
    """

    def fit(self, X, y=None):
        X = to_dataframe(X)
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return (to_dataframe(X) - self.mean_) / self.std_

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        return self.std_ * to_dataframe(X) + self.mean_


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    """Column-wise min–max scaling to [0, 1] preserving DataFrame structure.

    Constant columns are left unscaled.
    """

    def fit(self, X, y=None):
        X = to_dataframe(X)
        self.min_ = X.min()
        self.max_ = X.max()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        denom = (self.max_ - self.min_).replace(0, 1.0)
        return (to_dataframe(X) - self.min_) / denom

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        return (self.max_ - self.min_) * to_dataframe(X) + self.min_

