from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("auto_lab2/OpenML/data")
    index_path: Path = Path("auto_lab2/OpenML/data.csv")
    out_dir: Path = Path("auto_lab2/outputs")
    min_datasets: int = 300
    max_datasets: int = 330
    max_rows_per_dataset: int = 1200
    min_rows_per_dataset: int = 40
    max_features_for_corr: int = 40
    test_size: float = 0.3
    seed: int = 42
    pymfe_groups: tuple[str, ...] = ("general", "info-theory")
