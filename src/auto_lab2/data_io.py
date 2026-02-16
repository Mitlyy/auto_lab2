from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


@dataclass
class PreparedDataset:
    dataset_id: int
    dataset_name: str
    target_column: str
    X: pd.DataFrame
    y: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def load_openml_index(index_path: Path) -> dict[int, dict[str, str]]:
    if not index_path.exists():
        return {}
    index_df = pd.read_csv(index_path)
    out: dict[int, dict[str, str]] = {}
    for _, row in index_df.iterrows():
        dataset_id = int(row["id"])
        out[dataset_id] = {
            "name": str(row.get("name", dataset_id)),
            "target": str(row.get("target", "")),
        }
    return out


def list_dataset_paths(data_dir: Path) -> list[Path]:
    paths = [p for p in data_dir.glob("*.arff") if p.is_file()]

    def key_fn(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem.isdigit():
            return int(stem), path.name
        return 10**12, path.name

    return sorted(paths, key=key_fn)


def _decode_bytes(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def _decode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_decode_bytes)
            out[col] = out[col].replace("?", np.nan)
    return out


def _pick_target_column(
    columns: list[str],
    dataset_id: int,
    index_map: dict[int, dict[str, str]],
) -> str:
    lowered = {col.lower(): col for col in columns}
    candidates: list[str] = []
    if dataset_id in index_map:
        target_candidate = index_map[dataset_id].get("target", "")
        if target_candidate:
            candidates.append(target_candidate)
    candidates.extend(["class", "target", "label", "binaryClass"])
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return columns[-1]


def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    if non_missing == 0:
        return True
    return (int(converted.notna().sum()) / non_missing) >= 0.98


def _stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    indices = np.arange(len(X))
    class_counts = y.value_counts()
    can_stratify = y.nunique(dropna=True) > 1 and int(class_counts.min()) >= 2
    stratify = y if can_stratify else None
    try:
        selected_idx, _ = train_test_split(
            indices,
            train_size=max_rows,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(indices, size=max_rows, replace=False)

    selected_idx = np.sort(selected_idx)
    return (
        X.iloc[selected_idx].reset_index(drop=True),
        y.iloc[selected_idx].reset_index(drop=True),
    )


def prepare_dataset(
    path: Path,
    index_map: dict[int, dict[str, str]],
    max_rows: int,
    min_rows: int,
    seed: int,
) -> PreparedDataset:
    dataset_id = int(path.stem) if path.stem.isdigit() else -1

    raw_data, _ = arff.loadarff(path)
    df = pd.DataFrame(raw_data)
    if df.empty:
        raise ValueError("empty dataframe")
    df = _decode_dataframe(df)

    target_column = _pick_target_column(df.columns.tolist(), dataset_id, index_map)
    y = df[target_column].copy()
    y = y.replace("?", np.nan)
    valid_mask = y.notna()
    if int(valid_mask.sum()) < min_rows:
        raise ValueError("too few rows after target cleanup")

    X = df.loc[valid_mask, df.columns != target_column].copy()
    y = y.loc[valid_mask].reset_index(drop=True)
    X = X.reset_index(drop=True)
    if X.shape[1] == 0:
        raise ValueError("no feature columns")

    if len(X) > max_rows:
        X, y = _stratified_subsample(X, y, max_rows=max_rows, seed=seed)

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in X.columns:
        series = X[col]
        if _is_numeric_like(series):
            X[col] = pd.to_numeric(series, errors="coerce").astype(float)
            numeric_columns.append(col)
        else:
            X[col] = series.astype("object")
            categorical_columns.append(col)

    y = y.astype(str)
    class_counts = y.value_counts()
    if y.nunique(dropna=True) < 2:
        raise ValueError("single-class target")
    if int(class_counts.min()) < 2:
        raise ValueError("class with <2 instances")
    if len(X) < min_rows:
        raise ValueError("too few rows")

    dataset_name = index_map.get(dataset_id, {}).get("name", path.stem)
    return PreparedDataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        target_column=target_column,
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
