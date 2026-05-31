from collections.abc import Hashable
from dataclasses import dataclass

import narwhals as nw
import numpy as np
import pandas as pd

from narwhals.typing import IntoDataFrame


@dataclass(frozen=True)
class IndexMeta:
    """A pandas index's name and frequency, kept so they survive promotion to a column.

    ``None`` for indexless backends like polars.

    Parameters
    ----------
    name : hashable, optional
        The original index name. Default None.
    freq : pandas offset, optional
        The original ``DatetimeIndex.freq``, or None for an irregular index. Default None.
    """

    name: Hashable | None = None
    freq: object = None

    @classmethod
    def from_index(cls, index: pd.Index) -> "IndexMeta":
        return cls(name=index.name, freq=getattr(index, "freq", None))


def resolve_date_axis(df: nw.DataFrame, target: str | None) -> tuple[str, bool]:
    """Return ``(name, from_index)`` for *df*'s one datetime axis.

    ``from_index`` flags a pandas index rather than a column. Raise ``ValueError``,
    naming the offenders, if there are several datetime axes, none, or if *target* names
    a non-datetime column.
    """
    candidates: list[tuple[str, bool]] = [
        (name, False) for name, dtype in df.schema.items() if dtype.is_temporal()
    ]
    index = nw.maybe_get_index(df)
    if index is not None and pd.api.types.is_datetime64_any_dtype(index):
        candidates.append((index.name or "index", True))

    if len(candidates) > 1:
        found = [name for name, _ in candidates]
        raise ValueError(
            f"A time series frame must have exactly one datetime axis, but found {found}. "
            "Drop all but the one you want as the time axis."
        )
    if target is not None:
        if candidates and candidates[0][0] == target:
            return candidates[0]
        raise ValueError(f"date_column {target!r} is not a datetime column or index in the input.")
    if candidates:
        return candidates[0]
    raise ValueError(
        "No datetime column or index found. Pass date_column=... to name the time axis."
    )


def to_dated_frame(
    X: IntoDataFrame, target: str | None
) -> tuple[nw.DataFrame, str, IndexMeta | None]:
    """Wrap *X*, promoting a datetime index to a column.

    Return ``(df, name, index_meta)``: the narwhals frame, its datetime-axis column name,
    and the :class:`IndexMeta` if the axis came from a pandas index (else ``None``).
    """
    df = nw.from_native(X, eager_only=True)
    name, from_index = resolve_date_axis(df, target)
    if not from_index:
        return df, name, None
    index = nw.maybe_get_index(df)
    assert index is not None  # from_index is only set for a (non-None) datetime index
    index_meta = IndexMeta.from_index(index)
    df = df.with_columns(nw.new_series(name, index.to_numpy(), backend=df.implementation))
    df = nw.maybe_reset_index(df)
    return df, name, index_meta


def restore_frame(df: nw.DataFrame, name: str, index_meta: IndexMeta | None):
    """Convert *df* to native, restoring *name* to the index (name + freq) per *index_meta*."""
    if index_meta is None:
        return nw.to_native(df)
    native = nw.to_native(nw.maybe_set_index(df, column_names=[name]))
    native.index.name = index_meta.name
    if index_meta.freq is not None:
        native.index.freq = index_meta.freq
    return native


def replace_columns(df: nw.DataFrame, names: list[str], values: np.ndarray) -> nw.DataFrame:
    """Replace each column in *names* with the matching column of array *values*."""
    series = [
        nw.new_series(name, values[:, i], backend=df.implementation) for i, name in enumerate(names)
    ]
    return df.with_columns(*series)


def align_by_date(
    stored_dates: np.ndarray, stored_values: np.ndarray, query_dates: np.ndarray
) -> np.ndarray:
    """Reindex *stored_values* onto *query_dates* by date key, NaN where unmatched.

    The numpy form of the join-on-key inverse, keyed on *stored_dates*.
    """
    position = {date: i for i, date in enumerate(stored_dates)}
    out = np.full((len(query_dates), stored_values.shape[1]), np.nan)
    for j, date in enumerate(query_dates):
        i = position.get(date)
        if i is not None:
            out[j] = stored_values[i]
    return out
