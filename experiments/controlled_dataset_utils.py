from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from moe.data import FEATURES, TARGET, TIME

BASE_REQUIRED_COLUMNS = [TARGET, "avg_power", *FEATURES]
GPU_EXTRA_COLUMNS = [
    "delta_gpu_utilization",
    "delta_gpu_memory",
    "delta_gpu_power",
    "gpu_like_signal",
]
NORMAL_KINDS = ("cpu", "mem", "io", "net", "mixed")
RUN_RE = re.compile(r"^controlled-(cpu|mem|io|net|mixed)-(\d{2})$")
GPU_RUN_RE = re.compile(r"^controlled-gpu-like-(\d{2})$")
SIMULATION_NOTE = (
    "This is simulated GPU-like data for automatic expert expansion MVP, "
    "not real GPU measurement."
)


@dataclass(frozen=True)
class RunFingerprint:
    run_name: str
    parquet_sha256: str
    parquet_size: int
    parquet_rows: int
    parquet_columns: tuple[str, ...]
    collection_log_sha256: str
    collection_log_size: int
    metadata_exists: bool


@dataclass(frozen=True)
class VerificationSummary:
    parquet_count: int
    counts_by_kind: dict[str, int]
    summary_by_kind: pd.DataFrame


def runs_root(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return Path(__file__).resolve().parents[1] / "data" / "controlled_feature_moe_more_runs" / "runs"


def run_kind(run_name: str) -> str:
    m = RUN_RE.match(run_name)
    if m:
        return m.group(1)
    if GPU_RUN_RE.match(run_name):
        return "gpu-like"
    raise ValueError(f"Unrecognized controlled run name: {run_name}")


def parquet_path(run_dir: Path) -> Path:
    return run_dir / "datasets" / "process_interval_data.parquet"


def metadata_path(run_dir: Path) -> Path:
    return run_dir / "metadata.json"


def collection_log_path(run_dir: Path) -> Path:
    return run_dir / "collection.log"


def list_run_dirs(base_dir: str | Path | None = None) -> list[Path]:
    root = runs_root(base_dir)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_run_df(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path(run_dir))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_run(run_dir: Path) -> RunFingerprint:
    pq = parquet_path(run_dir)
    df = pd.read_parquet(pq)
    clog = collection_log_path(run_dir)
    return RunFingerprint(
        run_name=run_dir.name,
        parquet_sha256=file_sha256(pq),
        parquet_size=pq.stat().st_size,
        parquet_rows=len(df),
        parquet_columns=tuple(df.columns.tolist()),
        collection_log_sha256=file_sha256(clog),
        collection_log_size=clog.stat().st_size,
        metadata_exists=metadata_path(run_dir).exists(),
    )


def required_base_columns() -> list[str]:
    return [TIME, *BASE_REQUIRED_COLUMNS]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [col for col in required_columns if col not in df.columns]


def summarize_run(run_dir: Path, df: pd.DataFrame) -> dict[str, object]:
    return {
        "run_name": run_dir.name,
        "kind": run_kind(run_dir.name),
        "rows": int(len(df)),
        "intervals": int(df[TIME].nunique()) if TIME in df.columns else 0,
        "interval_energy_mean": float(df[TARGET].mean()),
        "interval_energy_std": float(df[TARGET].std(ddof=0)),
        "interval_energy_min": float(df[TARGET].min()),
        "interval_energy_max": float(df[TARGET].max()),
        "avg_power_mean": float(df["avg_power"].mean()),
        "avg_power_std": float(df["avg_power"].std(ddof=0)),
        "avg_power_min": float(df["avg_power"].min()),
        "avg_power_max": float(df["avg_power"].max()),
    }


def summarize_by_kind(run_summaries: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(run_summaries)
    grouped = (
        frame.groupby("kind", as_index=False)
        .agg(
            runs=("run_name", "count"),
            total_rows=("rows", "sum"),
            mean_rows=("rows", "mean"),
            total_intervals=("intervals", "sum"),
            mean_intervals=("intervals", "mean"),
            interval_energy_mean=("interval_energy_mean", "mean"),
            interval_energy_std=("interval_energy_std", "mean"),
            interval_energy_min=("interval_energy_min", "min"),
            interval_energy_max=("interval_energy_max", "max"),
            avg_power_mean=("avg_power_mean", "mean"),
            avg_power_std=("avg_power_std", "mean"),
            avg_power_min=("avg_power_min", "min"),
            avg_power_max=("avg_power_max", "max"),
        )
        .sort_values("kind")
        .reset_index(drop=True)
    )
    return grouped


def verify_existing_normal_runs(base_dir: str | Path | None = None) -> tuple[VerificationSummary, dict[str, RunFingerprint]]:
    root = runs_root(base_dir)
    run_dirs = [p for p in list_run_dirs(root) if RUN_RE.match(p.name)]
    parquets = [parquet_path(p) for p in run_dirs if parquet_path(p).exists()]
    if len(parquets) != 50:
        raise AssertionError(f"Expected exactly 50 process_interval_data.parquet files, found {len(parquets)}")

    counts = {kind: 0 for kind in NORMAL_KINDS}
    run_summaries: list[dict[str, object]] = []
    fingerprints: dict[str, RunFingerprint] = {}
    required = required_base_columns()

    for run_dir in run_dirs:
        kind = run_kind(run_dir.name)
        counts[kind] += 1
        df = load_run_df(run_dir)
        missing = validate_required_columns(df, required)
        if missing:
            raise AssertionError(f"Run {run_dir.name} is missing required columns: {missing}")
        run_summaries.append(summarize_run(run_dir, df))
        fingerprints[run_dir.name] = fingerprint_run(run_dir)

    for kind, count in counts.items():
        if count != 10:
            raise AssertionError(f"Expected exactly 10 runs for {kind}, found {count}")

    return VerificationSummary(
        parquet_count=len(parquets),
        counts_by_kind=counts,
        summary_by_kind=summarize_by_kind(run_summaries),
    ), fingerprints


def _normalize_signal(series: pd.Series) -> np.ndarray:
    arr = series.fillna(0.0).to_numpy(dtype=np.float64)
    arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    scale = arr.std()
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(arr)
    centered = arr - arr.mean()
    return centered / scale


def build_gpu_like_dataframe(source_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = source_df.copy()
    rng = np.random.default_rng(seed)

    cpu = _normalize_signal(df["delta_cpu_ns"])
    instr = _normalize_signal(df["delta_instructions"])
    mem = _normalize_signal(df["delta_rss_memory"])
    io = _normalize_signal(df["delta_io_bytes"])
    net = _normalize_signal(df["delta_net_send_bytes"])
    cycles = _normalize_signal(df["delta_cycles"])

    base_signal = 0.30 * cpu + 0.20 * instr + 0.22 * mem + 0.14 * io + 0.09 * net + 0.05 * cycles
    burst = 0.35 * np.sin(np.linspace(0.0, 6.0 * np.pi, len(df)))
    noise = rng.normal(0.0, 0.08, len(df))
    gpu_like_signal = np.clip(base_signal + burst + noise, -1.25, 3.5)
    gpu_like_signal = np.maximum(gpu_like_signal, 0.0)

    df["gpu_like_signal"] = gpu_like_signal.astype(np.float64)
    df["delta_gpu_utilization"] = np.clip(
        12.0 + 24.0 * gpu_like_signal + 6.0 * mem + 4.0 * cpu + rng.normal(0.0, 2.0, len(df)),
        0.0,
        100.0,
    )
    df["delta_gpu_memory"] = np.clip(
        64.0 + 420.0 * gpu_like_signal + 90.0 * np.maximum(mem, 0.0) + rng.normal(0.0, 18.0, len(df)),
        0.0,
        None,
    )
    df["delta_gpu_power"] = np.clip(
        18.0 + 32.0 * gpu_like_signal + 0.30 * df["delta_gpu_utilization"].to_numpy() + rng.normal(0.0, 1.8, len(df)),
        0.0,
        None,
    )

    base_energy = df[TARGET].fillna(0.0).to_numpy(dtype=np.float64)
    base_power = df["avg_power"].fillna(0.0).to_numpy(dtype=np.float64)
    gpu_energy_boost = 18.0 * gpu_like_signal + 0.55 * df["delta_gpu_power"].to_numpy(dtype=np.float64)
    gpu_power_boost = 8.0 * gpu_like_signal + 0.18 * df["delta_gpu_utilization"].to_numpy(dtype=np.float64)

    df[TARGET] = np.clip(base_energy + gpu_energy_boost + rng.normal(0.0, 2.0, len(df)), 1e-6, None)
    df["avg_power"] = np.clip(base_power + gpu_power_boost + rng.normal(0.0, 0.75, len(df)), 1e-6, None)
    return df


def ensure_gpu_run_absent_or_overwrite(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination run directory already exists and is not empty: {run_dir}. "
            "Pass overwrite=True to replace it."
        )


def write_gpu_run(run_dir: Path, df: pd.DataFrame, metadata: dict[str, object], collection_log: str, overwrite: bool) -> None:
    ensure_gpu_run_absent_or_overwrite(run_dir, overwrite)
    if run_dir.exists() and overwrite:
        for child in sorted(run_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        run_dir.rmdir()

    (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path(run_dir), index=False)
    metadata_path(run_dir).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    collection_log_path(run_dir).write_text(collection_log)


def make_gpu_run_metadata(run_name: str, source_run: str, seed: int, df: pd.DataFrame) -> dict[str, object]:
    return {
        "run_name": run_name,
        "workload_kind": "gpu-like",
        "simulated": True,
        "is_real_measurement": False,
        "purpose": "automatic expert expansion MVP",
        "description": SIMULATION_NOTE,
        "source_run": source_run,
        "seed": seed,
        "rows": int(len(df)),
        "added_columns": GPU_EXTRA_COLUMNS,
        "labels_recomputed": True,
        "label_dependency": {
            "interval_energy": "depends on gpu_like_signal and delta_gpu_power",
            "avg_power": "depends on gpu_like_signal and delta_gpu_utilization",
        },
    }


def make_gpu_collection_log(run_name: str, source_run: str, seed: int) -> str:
    return "\n".join([
        f"[INFO] run_name={run_name}",
        "[INFO] generated_from=synthetic_gpu_like_transformation",
        f"[INFO] source_run={source_run}",
        f"[INFO] seed={seed}",
        "[INFO] added_columns=delta_gpu_utilization,delta_gpu_memory,delta_gpu_power,gpu_like_signal",
        "[INFO] labels_recomputed=interval_energy,avg_power",
        f"[INFO] note={SIMULATION_NOTE}",
        "",
    ])


def verify_post_generation(base_dir: str | Path | None, original_fingerprints: dict[str, RunFingerprint]) -> VerificationSummary:
    root = runs_root(base_dir)
    run_dirs = list_run_dirs(root)
    parquets = [parquet_path(p) for p in run_dirs if parquet_path(p).exists()]
    if len(parquets) != 60:
        raise AssertionError(f"Expected total parquet count to be 60 after generation, found {len(parquets)}")

    counts = {kind: 0 for kind in (*NORMAL_KINDS, "gpu-like")}
    run_summaries: list[dict[str, object]] = []

    for run_dir in run_dirs:
        name = run_dir.name
        if RUN_RE.match(name):
            counts[run_kind(name)] += 1
            fp = fingerprint_run(run_dir)
            orig = original_fingerprints[name]
            if fp != orig:
                raise AssertionError(f"Original run changed unexpectedly: {name}")
            run_summaries.append(summarize_run(run_dir, load_run_df(run_dir)))
            continue

        if GPU_RUN_RE.match(name):
            counts["gpu-like"] += 1
            df = load_run_df(run_dir)
            missing = validate_required_columns(df, required_base_columns() + GPU_EXTRA_COLUMNS)
            if missing:
                raise AssertionError(f"GPU-like run {name} is missing required columns: {missing}")
            meta_file = metadata_path(run_dir)
            if not meta_file.exists():
                raise AssertionError(f"GPU-like run {name} is missing metadata.json")
            metadata = json.loads(meta_file.read_text())
            if not metadata.get("simulated") or metadata.get("is_real_measurement") is not False:
                raise AssertionError(f"GPU-like run {name} metadata must mark data as simulated and not real")
            description = json.dumps(metadata)
            if "automatic expert expansion MVP" not in description or "not real GPU measurement" not in description:
                raise AssertionError(f"GPU-like run {name} metadata is missing required provenance text")
            clog = collection_log_path(run_dir)
            if not clog.exists():
                raise AssertionError(f"GPU-like run {name} is missing collection.log")
            clog_text = clog.read_text()
            if "automatic expert expansion MVP" not in clog_text or "not real GPU measurement" not in clog_text:
                raise AssertionError(f"GPU-like run {name} collection.log is missing required provenance text")
            run_summaries.append(summarize_run(run_dir, df))

    for kind in NORMAL_KINDS:
        if counts[kind] != 10:
            raise AssertionError(f"Expected 10 runs for {kind} after generation, found {counts[kind]}")
    if counts["gpu-like"] != 10:
        raise AssertionError(f"Expected 10 gpu-like runs after generation, found {counts['gpu-like']}")

    return VerificationSummary(
        parquet_count=len(parquets),
        counts_by_kind=counts,
        summary_by_kind=summarize_by_kind(run_summaries),
    )
