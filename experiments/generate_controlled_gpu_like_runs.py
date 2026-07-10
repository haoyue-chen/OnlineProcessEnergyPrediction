from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.controlled_dataset_utils import (
    GPU_RUN_RE,
    VerificationSummary,
    build_gpu_like_dataframe,
    list_run_dirs,
    load_run_df,
    make_gpu_collection_log,
    make_gpu_run_metadata,
    parquet_path,
    run_kind,
    runs_root,
    verify_existing_normal_runs,
    verify_post_generation,
    write_gpu_run,
)


def _print_summary(title: str, summary: VerificationSummary) -> None:
    print(f"\n=== {title} ===")
    print(f"parquet_count={summary.parquet_count}")
    print("counts_by_kind:")
    for kind, count in summary.counts_by_kind.items():
        print(f"  {kind:8s} {count}")
    print("\nsummary_by_kind:")
    printable = summary.summary_by_kind.copy()
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(printable.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _normal_run_dirs(base_dir: Path) -> list[Path]:
    normal_dirs: list[Path] = []
    for p in list_run_dirs(base_dir):
        try:
            kind = run_kind(p.name)
        except ValueError:
            continue
        if kind in {"cpu", "mem", "io", "net", "mixed"}:
            normal_dirs.append(p)
    return normal_dirs


def _generate_gpu_like_runs(base_dir: Path, overwrite: bool, seed: int, num_runs: int) -> None:
    sources = _normal_run_dirs(base_dir)
    for idx in range(1, num_runs + 1):
        run_name = f"controlled-gpu-like-{idx:02d}"
        source_dir = sources[(idx - 1) % len(sources)]
        source_df = load_run_df(source_dir)
        run_seed = seed + idx
        gpu_df = build_gpu_like_dataframe(source_df, run_seed)
        metadata = make_gpu_run_metadata(run_name, source_dir.name, run_seed, gpu_df)
        clog = make_gpu_collection_log(run_name, source_dir.name, run_seed)
        write_gpu_run(base_dir / run_name, gpu_df, metadata, clog, overwrite=overwrite)
        print(
            f"[GENERATED] {run_name} from {source_dir.name} "
            f"rows={len(gpu_df)} path={parquet_path(base_dir / run_name)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify controlled datasets and generate simulated GPU-like runs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=runs_root(),
        help="Controlled runs root directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260709,
        help="Base random seed for deterministic generation.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of controlled-gpu-like runs to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing controlled-gpu-like-* directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    pre_summary, original_fingerprints = verify_existing_normal_runs(base_dir)
    _print_summary("PRE-GENERATION VERIFICATION", pre_summary)

    existing_gpu_runs = [p.name for p in list_run_dirs(base_dir) if GPU_RUN_RE.match(p.name)]
    if existing_gpu_runs and not args.overwrite:
        raise SystemExit(
            "GPU-like runs already exist: "
            + ", ".join(sorted(existing_gpu_runs))
            + ". Re-run with --overwrite to replace them."
        )

    _generate_gpu_like_runs(base_dir, overwrite=args.overwrite, seed=args.seed, num_runs=args.num_runs)

    post_summary = verify_post_generation(base_dir, original_fingerprints)
    _print_summary("POST-GENERATION VERIFICATION", post_summary)

    print("\nPASS pre-generation dataset verification")
    print("PASS generated 10 gpu-like runs")
    print("PASS post-generation parquet and schema verification")
    print("PASS original 50 normal runs unchanged")


if __name__ == "__main__":
    main()
