from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if np.all(~np.isfinite(values)):
        return 0.0
    return float(np.nanmean(values))


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    return numeric_columns, categorical_columns


def extract_custom_meta_features(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    max_features_for_corr: int = 40,
) -> dict[str, float]:
    if numeric_columns is None or categorical_columns is None:
        inferred_numeric, inferred_categorical = infer_column_types(X)
        numeric_columns = inferred_numeric
        categorical_columns = inferred_categorical

    n_samples = int(len(X))
    n_features = int(X.shape[1])
    n_numeric = int(len(numeric_columns))
    n_categorical = int(len(categorical_columns))
    n_classes = int(y.nunique(dropna=True))
    missing_ratio = float(X.isna().mean().mean()) if n_features else 0.0

    numeric_df = pd.DataFrame(index=X.index)
    for col in numeric_columns:
        series = pd.to_numeric(X[col], errors="coerce").astype(float)
        median = series.median()
        series = series.fillna(0.0 if pd.isna(median) else float(median))
        numeric_df[col] = series

    if not numeric_df.empty:
        means = numeric_df.mean(axis=0).to_numpy(dtype=float)
        stds = numeric_df.std(axis=0, ddof=1).to_numpy(dtype=float)
        skews = numeric_df.skew(axis=0).to_numpy(dtype=float)
        kurt = numeric_df.kurtosis(axis=0).to_numpy(dtype=float)
    else:
        means = np.array([], dtype=float)
        stds = np.array([], dtype=float)
        skews = np.array([], dtype=float)
        kurt = np.array([], dtype=float)

    num_mean_abs = _safe_mean(np.abs(means))
    num_std_mean = _safe_mean(stds)
    num_skew_abs_mean = _safe_mean(np.abs(skews))
    num_kurt_mean = _safe_mean(kurt)

    corr_abs_mean = 0.0
    if numeric_df.shape[1] > 1:
        corr_df = numeric_df.iloc[:, :max_features_for_corr].corr().abs()
        corr_values = corr_df.to_numpy(dtype=float)
        tri_idx = np.triu_indices_from(corr_values, k=1)
        corr_abs_mean = _safe_mean(corr_values[tri_idx])

    class_probs = y.value_counts(normalize=True).to_numpy(dtype=float)
    target_entropy = float(-np.sum(class_probs * np.log2(class_probs + 1e-12)))

    samples_per_feature = float(n_samples / max(n_features, 1))
    cat_to_num_ratio = float(n_categorical / max(n_numeric, 1))
    class_counts = y.value_counts()
    class_imbalance = float(class_counts.max() / class_counts.min()) if not class_counts.empty else 0.0

    unique_ratios = []
    for col in X.columns:
        nunique = int(X[col].nunique(dropna=True))
        unique_ratios.append(nunique / max(n_samples, 1))
    mean_unique_ratio = _safe_mean(np.asarray(unique_ratios, dtype=float))

    zero_ratio_numeric = float((numeric_df == 0).to_numpy(dtype=float).mean()) if not numeric_df.empty else 0.0

    cat_cardinality = []
    for col in categorical_columns:
        cat_cardinality.append(float(X[col].nunique(dropna=True)))
    avg_cat_cardinality = _safe_mean(np.asarray(cat_cardinality, dtype=float))

    return {
        "mf_basic_n_samples": float(n_samples),
        "mf_basic_n_features": float(n_features),
        "mf_basic_n_numeric": float(n_numeric),
        "mf_basic_n_categorical": float(n_categorical),
        "mf_basic_n_classes": float(n_classes),
        "mf_basic_missing_ratio": missing_ratio,
        "mf_stat_num_mean_abs": num_mean_abs,
        "mf_stat_num_std_mean": num_std_mean,
        "mf_stat_num_skew_abs_mean": num_skew_abs_mean,
        "mf_stat_num_kurt_mean": num_kurt_mean,
        "mf_stat_abs_corr_mean": corr_abs_mean,
        "mf_stat_target_entropy": target_entropy,
        "mf_struct_samples_per_feature": samples_per_feature,
        "mf_struct_cat_to_num_ratio": cat_to_num_ratio,
        "mf_struct_class_imbalance": class_imbalance,
        "mf_struct_mean_unique_ratio": mean_unique_ratio,
        "mf_struct_zero_ratio_numeric": zero_ratio_numeric,
        "mf_struct_avg_cat_cardinality": avg_cat_cardinality,
    }


def permute_dataset_invariant_view(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)

    row_idx = rng.permutation(len(X))
    X_perm = X.iloc[row_idx].reset_index(drop=True)
    y_perm = y.iloc[row_idx].reset_index(drop=True)

    col_order = X_perm.columns.to_list()
    rng.shuffle(col_order)
    X_perm = X_perm[col_order]

    for col in X_perm.columns:
        if pd.api.types.is_numeric_dtype(X_perm[col]):
            continue
        series = X_perm[col].copy()
        non_missing = series.dropna().astype(str)
        unique_values = non_missing.unique().tolist()
        shuffled = unique_values.copy()
        rng.shuffle(shuffled)
        mapping = dict(zip(unique_values, shuffled))
        X_perm[col] = series.astype(str).map(mapping).where(series.notna(), np.nan)

    y_non_missing = y_perm.dropna().astype(str)
    y_unique = y_non_missing.unique().tolist()
    y_shuffled = y_unique.copy()
    rng.shuffle(y_shuffled)
    y_mapping = dict(zip(y_unique, y_shuffled))
    y_perm = y_perm.astype(str).map(y_mapping).where(y_perm.notna(), np.nan)

    return X_perm, y_perm


def invariance_comparison(
    base_features: dict[str, float],
    permuted_features: dict[str, float],
) -> pd.DataFrame:
    keys = sorted(set(base_features) | set(permuted_features))
    rows = []
    for key in keys:
        base_value = float(base_features.get(key, 0.0))
        perm_value = float(permuted_features.get(key, 0.0))
        rows.append(
            {
                "feature": key,
                "base_value": base_value,
                "permuted_value": perm_value,
                "abs_diff": abs(base_value - perm_value),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)
