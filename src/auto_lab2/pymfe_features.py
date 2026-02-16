from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pymfe.mfe import MFE


def extract_pymfe_features(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    groups: Sequence[str],
    seed: int,
) -> dict[str, float]:
    frame = X.copy()
    for col in numeric_columns:
        series = pd.to_numeric(frame[col], errors="coerce").astype(float)
        median = series.median()
        series = series.fillna(0.0 if pd.isna(median) else float(median))
        frame[col] = series

    for col in categorical_columns:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

    cat_indices = [frame.columns.get_loc(col) for col in categorical_columns]
    target = y.fillna("__MISSING__").astype(str).to_numpy()

    mfe = MFE(groups=list(groups), summary=["mean"], random_state=seed)
    mfe.fit(
        frame.to_numpy(dtype=object),
        target,
        cat_cols=cat_indices,
        suppress_warnings=True,
    )
    names, values = mfe.extract(suppress_warnings=True)

    out: dict[str, float] = {}
    for name, value in zip(names, values):
        scalar = float(np.ravel(np.asarray(value))[0])
        if not np.isfinite(scalar):
            scalar = 0.0
        out[f"mf_pymfe_{name}"] = scalar
    return out
