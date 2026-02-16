from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_meta_projection(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    title: str,
    out_path: Path,
    seed: int,
) -> dict[str, float]:
    X = frame[feature_columns].fillna(0.0).to_numpy(dtype=float)
    y = frame[label_column].astype(str).to_numpy()

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=seed)
    projection = pca.fit_transform(X_scaled)

    unique_labels = sorted(np.unique(y).tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=140)
    for color, label in zip(colors, unique_labels):
        mask = y == label
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=26,
            alpha=0.85,
            color=color,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
    }


def plot_invariance_step3(
    invariance_df: pd.DataFrame,
    out_path: Path,
    title: str,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    base = invariance_df["base_value"].to_numpy(dtype=float)
    perm = invariance_df["permuted_value"].to_numpy(dtype=float)
    abs_diff = invariance_df["abs_diff"].to_numpy(dtype=float)

    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    unchanged_ratio = float(np.mean(abs_diff <= tolerance))

    global_min = float(min(np.min(base), np.min(perm)))
    global_max = float(max(np.max(base), np.max(perm)))
    if np.isclose(global_min, global_max):
        global_min -= 1.0
        global_max += 1.0

    top = invariance_df.sort_values("abs_diff", ascending=False).head(8).iloc[::-1].copy()
    positive = abs_diff[abs_diff > 0]
    floor_value = float(np.min(positive)) if positive.size else 1e-16
    top_plot_values = np.clip(top["abs_diff"].to_numpy(dtype=float), floor_value, None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    axes[0].scatter(base, perm, s=34, alpha=0.9, color="#2563eb")
    axes[0].plot(
        [global_min, global_max],
        [global_min, global_max],
        linestyle="--",
        color="#dc2626",
        linewidth=1.2,
    )
    axes[0].set_title("Before vs After permutation")
    axes[0].set_xlabel("Before permutation")
    axes[0].set_ylabel("After permutation")
    axes[0].grid(alpha=0.25)

    axes[1].barh(top["feature"], top_plot_values, color="#0f766e")
    axes[1].set_title("Top abs diff by meta-feature")
    axes[1].set_xlabel("|before - after|")
    axes[1].set_xscale("log")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"{title}\nmax_abs_diff={max_abs_diff:.3e}, mean_abs_diff={mean_abs_diff:.3e}, unchanged@{tolerance:.0e}={unchanged_ratio:.1%}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "unchanged_ratio_at_tolerance": unchanged_ratio,
    }
