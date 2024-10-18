from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DifferenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._x0 = None


    def fit(self, X, y=None):
        self.x0 = X[X.notna() & X.diff().isna()].copy()
        return self

    def transform(self, X, y=None):

        return X.diff()
    
    def inverse_transform(self, X, y=None):
        return X.fillna(self.x0).cumsum()
    
class DetrendTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, trend = 'c'):
        self.params = None
        self.trend = trend
        
    def _build_feature_matrix(self, X):
        trend = self.trend
        T = X.shape[0]
        features = None
        
        if 'c' in trend:
            features = np.ones(T)[:, None]
        if 't' in trend:
            lin_trend = np.arange(T)[:, None]
            features = lin_trend if features is None else np.c_[features, lin_trend]
        if 'tt' in trend:
            quad_trend = np.arange(T)[:, None] ** 2
            features = quad_trend if features is None else np.c_[features, quad_trend]
        
        return features
    
        
    def fit(self, X, y=None):
        features = self._build_feature_matrix(X)
        def regress(endog, exog):
            nan_mask = ~np.isnan(endog)
            return np.linalg.solve(exog[nan_mask].T @ exog[nan_mask], exog[nan_mask].T @ endog[nan_mask]) 
        
        params = np.apply_along_axis(regress, axis=0, arr=X.values, exog=features)
        self.params = np.atleast_1d(np.c_[params])
        
        return self
    
    def transform(self, X, y=None):
        n = 1 if len(X.shape) == 1 else X.shape[1]
        features = self._build_feature_matrix(X)
        X_hat = np.einsum('tkn, kn->tn', np.dstack([features] *  n), self.params)
        
        return X - X_hat
    
    def inverse_transform(self, X, y=None):
        # X are residuals (output of transform)
        n = 1 if len(X.shape) == 1 else X.shape[1]
        features = self._build_feature_matrix(X)
        X_hat = np.einsum('tkn, kn->tn', np.dstack([features] *  n), self.params)
        
        return X + X_hat
        
class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lamb=0):
        self.lamb = lamb
        self.signs = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        lamb = self.lamb
        
        if lamb == 0:
            return X.apply(np.log)
        
        return X.apply(lambda x: (np.sign(x) * np.abs(x) ** lamb - 1)  / lamb)
    
    def inverse_transform(self, X, y=None):
        lamb = self.lamb
        
        if lamb == 0:
            return X.apply(np.exp)
        
        return X.apply(lambda x: np.sign(lamb * x + 1) * np.abs(lamb * x + 1) ** (1 / lamb))
    
class PandasStandardScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.means = None
        self.stds = None
        
    def fit(self, X, y=None):
        self.means = X.mean()
        self.stds = X.std()
        
        return self
    
    def transform(self, X, y=None):
        means = self.means
        stds = self.stds
        
        return (X - means) / stds
    
    def inverse_transform(self, X, y=None):
        means = self.means
        stds = self.stds
        
        return stds * X + means
    
    
class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.mins = None
        self.maxes = None
        
    def fit(self, X, y=None):
        self.mins = X.min()
        self.maxes = X.max()
        
        return self
    
    def transform(self, X, y=None):
        mins = self.mins
        maxes = self.maxes
        
        return (X - mins) / (maxes - mins)
    
    def inverse_transform(self, X, y=None):
        mins  = self.mins
        maxes = self.maxes
        
        return (maxes - mins) * X + maxes