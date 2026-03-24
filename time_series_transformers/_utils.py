import pandas as pd


def to_dataframe(X: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return *X* as a :class:`~pandas.DataFrame` copy.

    A :class:`~pandas.Series` is promoted to a single-column DataFrame.
    """
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, pd.DataFrame):
        return X.copy()
    raise TypeError(f"Expected DataFrame or Series, got {type(X).__name__}.")

