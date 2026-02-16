from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data_io import PreparedDataset, list_dataset_paths, load_openml_index, prepare_dataset
from .meta_features import (
    extract_custom_meta_features,
    invariance_comparison,
    permute_dataset_invariant_view,
)
from .modeling import (
    BASE_ALGORITHMS,
    encode_features_for_models,
    encode_target,
    evaluate_base_algorithms,
    evaluate_meta_models,
    select_best_algorithm,
)
from .pymfe_features import extract_pymfe_features
from .visualization import plot_invariance_step3, plot_meta_projection


def _ensure_dirs(out_dir: Path) -> dict[str, Path]:
    custom_dir = out_dir / "custom"
    pymfe_dir = out_dir / "pymfe"
    for path in (out_dir, custom_dir, pymfe_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"out": out_dir, "custom": custom_dir, "pymfe": pymfe_dir}


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_custom_meta_dataset(
    config: ExperimentConfig,
    index_map: dict[int, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[PreparedDataset], pd.DataFrame]:
    records: list[dict[str, float | str | int]] = []
    score_records: list[dict[str, float | str | int]] = []
    datasets_used: list[PreparedDataset] = []
    skipped_rows: list[dict[str, str | int]] = []

    for path in list_dataset_paths(config.data_dir):
        if len(records) >= config.max_datasets:
            break

        dataset_id = int(path.stem) if path.stem.isdigit() else -1
        try:
            ds = prepare_dataset(
                path=path,
                index_map=index_map,
                max_rows=config.max_rows_per_dataset,
                min_rows=config.min_rows_per_dataset,
                seed=config.seed,
            )
            X_encoded = encode_features_for_models(ds.X, ds.numeric_columns, ds.categorical_columns)
            y_encoded = encode_target(ds.y)
            scores = evaluate_base_algorithms(
                X_encoded=X_encoded,
                y_encoded=y_encoded,
                seed=config.seed,
                test_size=config.test_size,
            )
            if not all(name in scores for name in BASE_ALGORITHMS):
                raise ValueError("failed to score all base algorithms")

            best_algorithm = select_best_algorithm(scores)
            custom_features = extract_custom_meta_features(
                X=ds.X,
                y=ds.y,
                numeric_columns=ds.numeric_columns,
                categorical_columns=ds.categorical_columns,
                max_features_for_corr=config.max_features_for_corr,
            )

            record = {
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "n_rows_used": int(len(ds.X)),
                "n_features_used": int(ds.X.shape[1]),
                "best_algorithm": best_algorithm,
            }
            record.update(custom_features)
            records.append(record)

            score_row = {
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "best_algorithm": best_algorithm,
            }
            score_row.update({f"score_{name}": float(value) for name, value in scores.items()})
            score_records.append(score_row)
            datasets_used.append(ds)

            if len(records) % 25 == 0:
                print(f"Processed datasets: {len(records)}")
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({"dataset_id": dataset_id, "file": path.name, "reason": str(exc)})

    if len(records) < config.min_datasets:
        raise RuntimeError(
            f"Only {len(records)} successful datasets, expected at least {config.min_datasets}"
        )

    custom_df = pd.DataFrame(records).sort_values("dataset_id").reset_index(drop=True)
    score_df = pd.DataFrame(score_records).sort_values("dataset_id").reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped_rows)
    return custom_df, score_df, skipped_df, datasets_used, pd.DataFrame(records)


def _compute_invariance_custom(
    ds: PreparedDataset,
    config: ExperimentConfig,
) -> pd.DataFrame:
    base = extract_custom_meta_features(
        X=ds.X,
        y=ds.y,
        numeric_columns=ds.numeric_columns,
        categorical_columns=ds.categorical_columns,
        max_features_for_corr=config.max_features_for_corr,
    )
    X_perm, y_perm = permute_dataset_invariant_view(ds.X, ds.y, seed=config.seed + 17)
    perm = extract_custom_meta_features(
        X=X_perm,
        y=y_perm,
        max_features_for_corr=config.max_features_for_corr,
    )
    return invariance_comparison(base, perm)


def _build_pymfe_meta_dataset(
    datasets_used: list[PreparedDataset],
    custom_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_map = {
        int(row["dataset_id"]): str(row["best_algorithm"])
        for _, row in custom_df[["dataset_id", "best_algorithm"]].iterrows()
    }

    rows: list[dict[str, float | str | int]] = []
    skipped_rows: list[dict[str, str | int]] = []

    for ds in datasets_used:
        try:
            features = extract_pymfe_features(
                X=ds.X,
                y=ds.y,
                numeric_columns=ds.numeric_columns,
                categorical_columns=ds.categorical_columns,
                groups=config.pymfe_groups,
                seed=config.seed,
            )
            row = {
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "n_rows_used": int(len(ds.X)),
                "n_features_used": int(ds.X.shape[1]),
                "best_algorithm": best_map[int(ds.dataset_id)],
            }
            row.update(features)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({"dataset_id": ds.dataset_id, "reason": str(exc)})

    pymfe_df = pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped_rows)

    if len(pymfe_df) < config.min_datasets:
        raise RuntimeError(
            f"PyMFE meta-dataset has only {len(pymfe_df)} rows, expected at least {config.min_datasets}"
        )

    first_dataset_id = int(pymfe_df.iloc[0]["dataset_id"])
    first_ds = next(ds for ds in datasets_used if ds.dataset_id == first_dataset_id)
    base = extract_pymfe_features(
        X=first_ds.X,
        y=first_ds.y,
        numeric_columns=first_ds.numeric_columns,
        categorical_columns=first_ds.categorical_columns,
        groups=config.pymfe_groups,
        seed=config.seed,
    )
    X_perm, y_perm = permute_dataset_invariant_view(first_ds.X, first_ds.y, seed=config.seed + 17)
    perm = extract_pymfe_features(
        X=X_perm,
        y=y_perm,
        numeric_columns=[c for c in X_perm.columns if c in first_ds.numeric_columns],
        categorical_columns=[c for c in X_perm.columns if c in first_ds.categorical_columns],
        groups=config.pymfe_groups,
        seed=config.seed,
    )
    invariance_df = invariance_comparison(base, perm)
    return pymfe_df, skipped_df, invariance_df


def _run_meta_level(
    frame: pd.DataFrame,
    feature_prefix: str,
    label_column: str,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [col for col in frame.columns if col.startswith(feature_prefix)]
    if not feature_columns:
        raise RuntimeError(f"No feature columns with prefix: {feature_prefix}")
    X_meta = frame[feature_columns].fillna(0.0)
    y_meta = frame[label_column].astype(str)
    metrics_df = evaluate_meta_models(X_meta, y_meta, seed=seed)
    return metrics_df, feature_columns


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    paths = _ensure_dirs(config.out_dir)
    index_map = load_openml_index(config.index_path)

    custom_df, score_df, skipped_custom, datasets_used, _ = _build_custom_meta_dataset(config, index_map)
    custom_df.to_csv(paths["custom"] / "meta_dataset_custom.csv", index=False)
    score_df.to_csv(paths["custom"] / "dataset_model_scores.csv", index=False)
    if not skipped_custom.empty:
        skipped_custom.to_csv(paths["custom"] / "skipped_datasets.csv", index=False)

    custom_invariance = _compute_invariance_custom(datasets_used[0], config=config)
    custom_invariance.to_csv(paths["custom"] / "invariance_custom.csv", index=False)
    custom_invariance_info = plot_invariance_step3(
        invariance_df=custom_invariance,
        out_path=paths["custom"] / "step3_invariance_custom.png",
        title="Step 3 check (custom meta-features): invariance to row/column/category permutations",
    )
    _save_json(custom_invariance_info, paths["custom"] / "invariance_custom_summary.json")

    custom_projection_info = plot_meta_projection(
        frame=custom_df,
        feature_columns=[col for col in custom_df.columns if col.startswith("mf_") and not col.startswith("mf_pymfe_")],
        label_column="best_algorithm",
        title="Custom meta-features: PCA projection",
        out_path=paths["custom"] / "meta_projection_custom.png",
        seed=config.seed,
    )
    _save_json(custom_projection_info, paths["custom"] / "projection_custom.json")

    custom_meta_metrics, custom_feature_columns = _run_meta_level(
        frame=custom_df,
        feature_prefix="mf_",
        label_column="best_algorithm",
        seed=config.seed,
    )
    custom_meta_metrics.to_csv(paths["custom"] / "meta_models_custom.csv", index=False)

    custom_meta_meta = extract_custom_meta_features(
        X=custom_df[custom_feature_columns],
        y=custom_df["best_algorithm"],
        numeric_columns=custom_feature_columns,
        categorical_columns=[],
        max_features_for_corr=config.max_features_for_corr,
    )
    _save_json(custom_meta_meta, paths["custom"] / "meta_meta_features_custom.json")

    pymfe_df, skipped_pymfe, pymfe_invariance = _build_pymfe_meta_dataset(
        datasets_used=datasets_used,
        custom_df=custom_df,
        config=config,
    )
    pymfe_df.to_csv(paths["pymfe"] / "meta_dataset_pymfe.csv", index=False)
    if not skipped_pymfe.empty:
        skipped_pymfe.to_csv(paths["pymfe"] / "skipped_pymfe.csv", index=False)
    pymfe_invariance.to_csv(paths["pymfe"] / "invariance_pymfe.csv", index=False)
    _save_json(
        {
            "max_abs_diff": float(pymfe_invariance["abs_diff"].max()),
            "mean_abs_diff": float(pymfe_invariance["abs_diff"].mean()),
        },
        paths["pymfe"] / "invariance_pymfe_summary.json",
    )

    pymfe_projection_info = plot_meta_projection(
        frame=pymfe_df,
        feature_columns=[col for col in pymfe_df.columns if col.startswith("mf_pymfe_")],
        label_column="best_algorithm",
        title="PyMFE meta-features: PCA projection",
        out_path=paths["pymfe"] / "meta_projection_pymfe.png",
        seed=config.seed,
    )
    _save_json(pymfe_projection_info, paths["pymfe"] / "projection_pymfe.json")

    pymfe_meta_metrics, pymfe_feature_columns = _run_meta_level(
        frame=pymfe_df,
        feature_prefix="mf_pymfe_",
        label_column="best_algorithm",
        seed=config.seed,
    )
    pymfe_meta_metrics.to_csv(paths["pymfe"] / "meta_models_pymfe.csv", index=False)

    pymfe_meta_meta = extract_custom_meta_features(
        X=pymfe_df[pymfe_feature_columns],
        y=pymfe_df["best_algorithm"],
        numeric_columns=pymfe_feature_columns,
        categorical_columns=[],
        max_features_for_corr=config.max_features_for_corr,
    )
    _save_json(pymfe_meta_meta, paths["pymfe"] / "meta_meta_features_pymfe.json")

    summary = {
        "n_datasets_custom": int(len(custom_df)),
        "n_datasets_pymfe": int(len(pymfe_df)),
        "base_algorithms": list(BASE_ALGORITHMS),
        "validation_metric": "balanced_accuracy",
        "custom_invariance_max_abs_diff": float(custom_invariance["abs_diff"].max()),
        "pymfe_invariance_max_abs_diff": float(pymfe_invariance["abs_diff"].max()),
        "custom_best_meta_model": custom_meta_metrics.iloc[0].to_dict(),
        "pymfe_best_meta_model": pymfe_meta_metrics.iloc[0].to_dict(),
    }
    _save_json(summary, paths["out"] / "report.json")

    return {
        "out_dir": paths["out"],
        "custom_meta_dataset": paths["custom"] / "meta_dataset_custom.csv",
        "pymfe_meta_dataset": paths["pymfe"] / "meta_dataset_pymfe.csv",
        "report_json": paths["out"] / "report.json",
    }
