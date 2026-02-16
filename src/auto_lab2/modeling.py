from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_ALGORITHMS = ("logreg", "knn", "random_forest")


def encode_features_for_models(
    X: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=X.index)

    for col in numeric_columns:
        series = pd.to_numeric(X[col], errors="coerce").astype(float)
        median = series.median()
        series = series.fillna(0.0 if pd.isna(median) else float(median))
        encoded[col] = series

    for col in categorical_columns:
        series = X[col].fillna("__MISSING__").astype(str)
        encoded[col] = pd.Categorical(series).codes.astype(float)

    if encoded.shape[1] == 0:
        raise ValueError("encoded feature matrix is empty")
    return encoded


def encode_target(y: pd.Series) -> np.ndarray:
    series = y.fillna("__MISSING__").astype(str)
    codes = pd.Categorical(series).codes
    return codes.astype(int)


def _build_base_models(seed: int) -> dict[str, object]:
    return {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            random_state=seed,
            n_jobs=1,
        ),
    }


def evaluate_base_algorithms(
    X_encoded: pd.DataFrame,
    y_encoded: np.ndarray,
    seed: int,
    test_size: float,
) -> dict[str, float]:
    if len(np.unique(y_encoded)) < 2:
        raise ValueError("target has <2 classes")

    bincount = np.bincount(y_encoded)
    can_stratify = bincount.size > 1 and bincount.min() >= 2
    stratify = y_encoded if can_stratify else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_encoded,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    scores: dict[str, float] = {}
    for name, model in _build_base_models(seed=seed).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        scores[name] = float(balanced_accuracy_score(y_valid, y_pred))
    return scores


def select_best_algorithm(scores: dict[str, float]) -> str:
    if not scores:
        raise ValueError("empty score dictionary")
    best_name, _ = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best_name


def evaluate_meta_models(
    X_meta: pd.DataFrame,
    y_meta: pd.Series,
    seed: int,
) -> pd.DataFrame:
    y_codes = pd.Categorical(y_meta.astype(str)).codes.astype(int)
    if len(np.unique(y_codes)) < 2:
        raise ValueError("meta-target has <2 classes")

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            random_state=seed,
            n_jobs=1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        cv_result = cross_validate(
            estimator=model,
            X=X_meta,
            y=y_codes,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
            },
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
                "accuracy_std": float(np.std(cv_result["test_accuracy"])),
                "balanced_accuracy_mean": float(np.mean(cv_result["test_balanced_accuracy"])),
                "balanced_accuracy_std": float(np.std(cv_result["test_balanced_accuracy"])),
            }
        )

    return pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)
