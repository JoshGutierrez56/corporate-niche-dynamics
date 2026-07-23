"""Transparent descriptive diagnostics for P3 component and axis tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def missingness_table(
    frame: pd.DataFrame, columns: Sequence[str], *, section: str
) -> pd.DataFrame:
    """Return counts and rates without imputing any observation."""

    records = []
    for column in columns:
        missing = int(frame[column].isna().sum())
        records.append(
            {
                "section": section,
                "variable": column,
                "rows": int(len(frame)),
                "nonmissing": int(len(frame) - missing),
                "missing": missing,
                "missing_rate": float(missing / len(frame)) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def correlation_long(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    section: str,
    method: str = "spearman",
) -> pd.DataFrame:
    """Return a long pairwise correlation table with pairwise sample counts."""

    records: list[dict[str, object]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index:]:
            valid = frame[left].notna() & frame[right].notna()
            if int(valid.sum()) >= 3:
                left_values = frame.loc[valid, left]
                right_values = frame.loc[valid, right]
                if method == "spearman":
                    left_values = left_values.rank(method="average")
                    right_values = right_values.rank(method="average")
                correlation = float(left_values.corr(right_values, method="pearson"))
            else:
                correlation = np.nan
            records.append(
                {
                    "section": section,
                    "left": left,
                    "right": right,
                    "method": method,
                    "observations": int(valid.sum()),
                    "correlation": correlation,
                }
            )
    return pd.DataFrame.from_records(records)


def vif_table(
    frame: pd.DataFrame, columns: Sequence[str], *, section: str
) -> pd.DataFrame:
    """Compute correlation-matrix VIFs as a descriptive collinearity diagnostic."""

    complete = frame[list(columns)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(complete) < max(10, len(columns) + 2):
        return pd.DataFrame(
            {
                "section": section,
                "variable": list(columns),
                "observations": len(complete),
                "vif": np.nan,
            }
        )
    standardized = complete.subtract(complete.mean()).divide(
        complete.std(ddof=0).replace(0.0, np.nan)
    )
    usable = [column for column in columns if standardized[column].notna().all()]
    if len(usable) < 2:
        return pd.DataFrame(
            {
                "section": section,
                "variable": list(columns),
                "observations": len(complete),
                "vif": np.nan,
            }
        )
    correlation = standardized[usable].corr().to_numpy(dtype=float)
    inverse = np.linalg.pinv(correlation, hermitian=True)
    values = dict(zip(usable, np.diag(inverse), strict=True))
    return pd.DataFrame(
        {
            "section": section,
            "variable": list(columns),
            "observations": len(complete),
            "vif": [float(values.get(column, np.nan)) for column in columns],
        }
    )


def count_nonfinite(frame: pd.DataFrame, columns: Sequence[str]) -> int:
    """Count positive or negative infinity while allowing declared missingness."""

    values = frame[list(columns)].select_dtypes(include=[np.number]).to_numpy()
    return int(np.isinf(values).sum())
