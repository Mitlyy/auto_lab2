from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_lab2.config import ExperimentConfig
from auto_lab2.pipeline import run_experiment


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Meta-learning pipeline for OpenML ARFF datasets")
    parser.add_argument("--data-dir", type=str, default="auto_lab2/OpenML/data")
    parser.add_argument("--index-path", type=str, default="auto_lab2/OpenML/data.csv")
    parser.add_argument("--out-dir", type=str, default="auto_lab2/outputs")
    parser.add_argument("--min-datasets", type=int, default=300, help="Required successful datasets")
    parser.add_argument("--max-datasets", type=int, default=330, help="Upper limit of processed datasets")
    parser.add_argument("--max-rows", type=int, default=1200, help="Row cap per dataset")
    parser.add_argument("--test-size", type=float, default=0.3, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return ExperimentConfig(
        data_dir=Path(args.data_dir),
        index_path=Path(args.index_path),
        out_dir=Path(args.out_dir),
        min_datasets=args.min_datasets,
        max_datasets=args.max_datasets,
        max_rows_per_dataset=args.max_rows,
        test_size=args.test_size,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    artifacts = run_experiment(config)
    print(f"Artifacts saved to: {artifacts['out_dir']}")
    print(f"Custom meta-dataset: {artifacts['custom_meta_dataset']}")
    print(f"PyMFE meta-dataset: {artifacts['pymfe_meta_dataset']}")
    print(f"Summary report: {artifacts['report_json']}")


if __name__ == "__main__":
    main()
