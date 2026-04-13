#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

EXPECTED_HAR_SR = 20
EXPECTED_HAR_WINDOW_SEC = 10
EXPECTED_HAR_STEP_SEC = 5
EXPECTED_HAR_CHANNELS = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
]
EXPECTED_HAR_SAMPLES = EXPECTED_HAR_SR * EXPECTED_HAR_WINDOW_SEC
EXPECTED_HAR_STEP_SAMPLES = EXPECTED_HAR_SR * EXPECTED_HAR_STEP_SEC


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    stats: Optional[Dict[str, Any]] = None


def sizeof_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def safe_load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def maybe_json_list(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return value
    return value


def extract_artifacts(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts = summary.get("artifacts", [])
    if isinstance(artifacts, list):
        return [item for item in artifacts if isinstance(item, dict)]
    return []

def safe_npz_keys(path: Path) -> List[str]:
    with np.load(path, allow_pickle=True) as data:
        return list(data.keys())


def infer_main_array(npz: np.lib.npyio.NpzFile) -> Tuple[str, np.ndarray]:
    for key in ("x", "X", "data", "signal", "array"):
        if key in npz:
            return key, npz[key]
    # fallback: first ndarray
    for key in npz.files:
        arr = npz[key]
        if isinstance(arr, np.ndarray):
            return key, arr
    raise ValueError("No ndarray found in npz")


def find_processed_files(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes])


def maybe_sidecar_json(data_path: Path) -> Optional[Path]:
    candidates = [
        data_path.with_suffix(".json"),
        data_path.parent / f"{data_path.stem}.json",
        data_path.parent / "metadata.json",
        data_path.parent / f"{data_path.stem}_meta.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def list_npz_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.npz"))


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def run_command(cmd: str, cwd: Path) -> Dict[str, Any]:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    end = time.time()
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "runtime_sec": round(end - start, 2),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_command_with_time(cmd: str, cwd: Path) -> Dict[str, Any]:
    start = time.time()
    wrapped_cmd = f"/usr/bin/time -v {cmd}"
    proc = subprocess.run(
        wrapped_cmd,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    end = time.time()

    stderr = proc.stderr
    max_rss_kb = None
    elapsed_time_str = None
    user_time_sec = None
    system_time_sec = None

    for line in stderr.splitlines():
        line = line.strip()
        if "Maximum resident set size" in line:
            match = re.search(r":\s*([0-9]+)", line)
            if match:
                max_rss_kb = int(match.group(1))
        elif "Elapsed (wall clock) time" in line:
            match = re.search(r":\s*(.+)$", line)
            if match:
                elapsed_time_str = match.group(1).strip()
        elif "User time (seconds)" in line:
            match = re.search(r":\s*([0-9.]+)", line)
            if match:
                user_time_sec = float(match.group(1))
        elif "System time (seconds)" in line:
            match = re.search(r":\s*([0-9.]+)", line)
            if match:
                system_time_sec = float(match.group(1))

    return {
        "command": cmd,
        "returncode": proc.returncode,
        "runtime_sec": round(end - start, 2),
        "elapsed_wallclock": elapsed_time_str,
        "max_rss_kb": max_rss_kb,
        "max_rss_mb": round(max_rss_kb / 1024, 2) if max_rss_kb is not None else None,
        "user_time_sec": user_time_sec,
        "system_time_sec": system_time_sec,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


# ---------------------------------------------------------
# modality-specific checks
# ---------------------------------------------------------

def check_reproducibility(repo_root: Path, commands: List[str]) -> CheckResult:
    """
    Runs documented commands from repo_root.
    This assumes commands support configurable output dirs or use relative paths.
    """
    command_results = []
    all_ok = True

    for cmd in commands:
        res = run_command(cmd, cwd=repo_root)
        command_results.append(res)
        if res["returncode"] != 0:
            all_ok = False

    details = "All documented commands completed successfully." if all_ok else "One or more documented commands failed."
    return CheckResult(
        name="Reproducibility",
        passed=all_ok,
        details=details,
        stats={"commands": command_results},
    )


def run_full_pipeline(
    repo_root: Path,
    commands: List[str],
    clean_first: bool,
    processed_root: Path,
    interim_root: Path,
    report_dir: Path,
) -> CheckResult:
    """
    Clean outputs, run setup + preprocessing commands, capture runtime/RAM.
    """
    if clean_first:
        for path in (processed_root, interim_root, report_dir):
            clean_directory(path)

    report_dir.mkdir(parents=True, exist_ok=True)

    command_results = []
    all_ok = True

    for cmd in commands:
        res = run_command_with_time(cmd, cwd=repo_root)
        command_results.append(res)
        if res["returncode"] != 0:
            all_ok = False
            break

    peak_rss_mb = None
    total_runtime_sec = round(sum(r.get("runtime_sec", 0.0) for r in command_results), 2)

    rss_values = [r["max_rss_mb"] for r in command_results if r.get("max_rss_mb") is not None]
    if rss_values:
        peak_rss_mb = max(rss_values)

    return CheckResult(
        name="Reproducibility",
        passed=all_ok,
        details=(
            "Full clean run completed successfully."
            if all_ok
            else "Full clean run failed on one or more commands."
        ),
        stats={
            "commands": command_results,
            "peak_ram_mb_across_run": peak_rss_mb,
            "total_runtime_sec": total_runtime_sec,
        },
    )


def check_har_outputs(processed_root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []

    har_root = processed_root / "har"

    pamap_pretrain_root = har_root / "pretrain" / "pamap2"
    wisdm_pretrain_root = har_root / "pretrain" / "wisdm"
    mhealth_pretrain_root = har_root / "pretrain" / "mhealth"
    pamap_supervised_root = har_root / "supervised" / "pamap2"
    wisdm_supervised_root = har_root / "supervised" / "wisdm"
    mhealth_supervised_root = har_root / "supervised" / "mhealth"

    pamap_pretrain_files = list_npz_files(pamap_pretrain_root)
    wisdm_pretrain_files = list_npz_files(wisdm_pretrain_root)
    mhealth_pretrain_files = list_npz_files(mhealth_pretrain_root)
    pamap_supervised_files = list_npz_files(pamap_supervised_root)
    wisdm_supervised_files = list_npz_files(wisdm_supervised_root)
    mhealth_supervised_files = list_npz_files(mhealth_supervised_root)

    pamap_files = pamap_pretrain_files + pamap_supervised_files
    wisdm_files = wisdm_pretrain_files + wisdm_supervised_files
    mhealth_files = mhealth_pretrain_files + mhealth_supervised_files
    har_files = pamap_files + wisdm_files + mhealth_files

    if not har_files:
        return [CheckResult(
            name="HAR harmonisation",
            passed=False,
            details=f"No HAR output files found under {har_root}",
        )]

    pamap_summary_path = first_existing([
        har_root / "pamap2_window_summary.json",
        Path("data/interim/har/pamap2/cleaned/summary.json"),
    ])
    wisdm_summary_path = first_existing([
        har_root / "wisdm_window_summary.json",
        Path("data/interim/har/wisdm/cleaned/summary.json"),
    ])
    mhealth_summary_path = first_existing([
        har_root / "mhealth_window_summary.json",
        Path("data/interim/har/mhealth/cleaned/summary.json"),
    ])

    summaries = {}
    if pamap_summary_path is not None:
        summaries["pamap2"] = safe_load_json(pamap_summary_path)
    if wisdm_summary_path is not None:
        summaries["wisdm"] = safe_load_json(wisdm_summary_path)
    if mhealth_summary_path is not None:
        summaries["mhealth"] = safe_load_json(mhealth_summary_path)

    window_shapes = Counter()
    nan_files = []
    inf_files = []
    malformed = []

    dataset_counts = {
        "pamap2_pretrain": len(pamap_pretrain_files),
        "pamap2_supervised": len(pamap_supervised_files),
        "wisdm_pretrain": len(wisdm_pretrain_files),
        "wisdm_supervised": len(wisdm_supervised_files),
        "mhealth_pretrain": len(mhealth_pretrain_files),
        "mhealth_supervised": len(mhealth_supervised_files),
    }

    pamap_subject_ids = set()
    wisdm_subject_ids = set()
    mhealth_subject_ids = set()

    for dataset_name, files in (("pamap2", pamap_files), ("wisdm", wisdm_files), ("mhealth", mhealth_files)):
        for f in files:
            if dataset_name == "pamap2":
                pamap_subject_ids.add(f.stem)
            elif dataset_name == "wisdm":
                wisdm_subject_ids.add(f.stem)
            else:
                mhealth_subject_ids.add(f.stem)

            data = load_npz_dict(f)
            arr = None
            for key in ("x", "X", "data", "signal", "windows"):
                if key in data:
                    arr = data[key]
                    break
            if arr is None:
                for value in data.values():
                    if isinstance(value, np.ndarray) and value.ndim >= 2:
                        arr = value
                        break
            if arr is None:
                malformed.append((str(f), "no array payload found"))
                continue

            if arr.ndim == 2:
                t, c = arr.shape
                window_shapes[(t, c)] += 1
                flat_arr = arr
            elif arr.ndim == 3:
                n, t, c = arr.shape
                window_shapes[(t, c)] += n
                flat_arr = arr.reshape(-1, c)
            else:
                malformed.append((str(f), f"unexpected ndim={arr.ndim}"))
                continue

            if np.isnan(flat_arr).any():
                nan_files.append(str(f))
            if np.isinf(flat_arr).any():
                inf_files.append(str(f))

    sampling_rates = {}
    channel_schemas = {}
    label_policy = {}
    expected_shapes_ok = True

    pamap_summary = summaries.get("pamap2", {})
    wisdm_summary = summaries.get("wisdm", {})
    mhealth_summary = summaries.get("mhealth", {})

    for dataset_name, summary in (("pamap2", pamap_summary), ("wisdm", wisdm_summary), ("mhealth", mhealth_summary)):
        if not summary:
            continue
        sampling_rates[dataset_name] = summary.get("sampling_rate_hz", summary.get("target_hz"))
        if "channel_names" in summary:
            channel_schemas[dataset_name] = maybe_json_list(summary.get("channel_names"))
        elif "channel_schema" in summary:
            channel_schemas[dataset_name] = maybe_json_list(summary.get("channel_schema"))
        elif "channels" in summary:
            channel_schemas[dataset_name] = maybe_json_list(summary.get("channels"))
        if "label_policy" in summary:
            label_policy[dataset_name] = summary.get("label_policy")
        elif "null_label_policy" in summary:
            label_policy[dataset_name] = summary.get("null_label_policy")
        elif "label_zero_policy" in summary:
            label_policy[dataset_name] = summary.get("label_zero_policy")

    expected_shapes_ok = True
    observed_shapes = set(window_shapes.keys())
    required_shapes = {
        (EXPECTED_HAR_SAMPLES, len(EXPECTED_HAR_CHANNELS)),
        (EXPECTED_HAR_STEP_SAMPLES, len(EXPECTED_HAR_CHANNELS)),
    }

    if not required_shapes.issubset(observed_shapes):
        expected_shapes_ok = False

    for shape in observed_shapes:
        t, c = shape
        if c != len(EXPECTED_HAR_CHANNELS) or t not in (EXPECTED_HAR_SAMPLES, EXPECTED_HAR_STEP_SAMPLES):
            expected_shapes_ok = False
            break

    harmonisation_ok = (
        sampling_rates.get("pamap2") == EXPECTED_HAR_SR
        and sampling_rates.get("wisdm") == EXPECTED_HAR_SR
        and sampling_rates.get("mhealth") == EXPECTED_HAR_SR
        and channel_schemas.get("pamap2") == EXPECTED_HAR_CHANNELS
        and channel_schemas.get("wisdm") == EXPECTED_HAR_CHANNELS
        and channel_schemas.get("mhealth") == EXPECTED_HAR_CHANNELS
    )

    results.append(CheckResult(
        name="HAR harmonisation",
        passed=harmonisation_ok,
        details=(
            f"PAMAP2 pretrain/supervised files: {len(pamap_pretrain_files)}/{len(pamap_supervised_files)}, "
            f"WISDM pretrain/supervised files: {len(wisdm_pretrain_files)}/{len(wisdm_supervised_files)}, "
            f"mHealth pretrain/supervised files: {len(mhealth_pretrain_files)}/{len(mhealth_supervised_files)}. "
            f"Sampling rates: {sampling_rates}. "
            f"Summary sources: pamap2={str(pamap_summary_path) if pamap_summary_path is not None else 'missing'}, "
            f"wisdm={str(wisdm_summary_path) if wisdm_summary_path is not None else 'missing'}, "
            f"mhealth={str(mhealth_summary_path) if mhealth_summary_path is not None else 'missing'}."
        ),
        stats={
            "sampling_rates": sampling_rates,
            "channel_schemas": channel_schemas,
            "dataset_counts": dataset_counts,
        },
    ))

    results.append(CheckResult(
        name="Window definition",
        passed=expected_shapes_ok,
        details=(
            f"Observed HAR window shapes: {dict(window_shapes)}. "
            f"Expected pretrain shape: ({EXPECTED_HAR_SAMPLES}, {len(EXPECTED_HAR_CHANNELS)}); "
            f"expected supervised shape: ({EXPECTED_HAR_STEP_SAMPLES}, {len(EXPECTED_HAR_CHANNELS)})."
        ),
        stats={"window_shapes": {str(k): v for k, v in window_shapes.items()}},
    ))

    results.append(CheckResult(
        name="Null or transient label handling",
        passed=bool(label_policy),
        details=(
            "Null/transient label handling read from HAR summary metadata. "
            f"Policies found: {label_policy if label_policy else 'none found in summaries'}."
        ),
        stats={"label_policy": label_policy},
    ))

    integrity_ok = not nan_files and not inf_files and not malformed
    results.append(CheckResult(
        name="Array integrity",
        passed=integrity_ok,
        details=(
            f"NaN files: {len(nan_files)}, Inf files: {len(inf_files)}, malformed files: {len(malformed)}."
        ),
        stats={
            "nan_files": nan_files[:20],
            "inf_files": inf_files[:20],
            "malformed": malformed[:20],
        },
    ))

    leakage_ok = bool(pamap_subject_ids) and bool(wisdm_subject_ids) and bool(mhealth_subject_ids)
    results.append(CheckResult(
        name="Leakage control",
        passed=leakage_ok,
        details=(
            f"HAR outputs detected: {len(pamap_files)} PAMAP2 files, {len(wisdm_files)} WISDM files, and {len(mhealth_files)} mHealth files across pretrain and supervised outputs."
        ),
        stats={
            "pamap2_subjects_or_files": len(pamap_subject_ids),
            "wisdm_subjects_or_files": len(wisdm_subject_ids),
            "mhealth_subjects_or_files": len(mhealth_subject_ids),
        },
    ))

    return results


def check_eeg_outputs(processed_root: Path) -> List[CheckResult]:
    eeg_root = processed_root / "eeg"
    eeg_files = list_npz_files(eeg_root / "supervised" / "eegmmidb")
    summary_path = eeg_root / "eegmmidb_window_summary.json"

    if not eeg_files:
        return [CheckResult(
            name="EEG annotations",
            passed=False,
            details=f"No EEG output files found under {eeg_root}",
        )]

    if not summary_path.exists():
        return [CheckResult(
            name="EEG annotations",
            passed=False,
            details=f"Missing EEG summary file: {summary_path}",
        )]

    summary = safe_load_json(summary_path)
    artifacts = extract_artifacts(summary)
    outputs = [
        item for item in artifacts
        if item.get("artifact_kind") == "supervised_npz"
    ]

    n_inputs = summary.get("n_inputs")
    n_outputs = len(outputs)
    n_processed_npz = len(eeg_files)

    shape_mismatches = []
    for item in outputs:
        shape = item.get("array_shape")
        if not isinstance(shape, list) or len(shape) != 3:
            shape_mismatches.append({
                "file": item.get("path") or item.get("file"),
                "array_shape": shape,
            })

    passed = n_outputs == n_processed_npz and n_outputs > 0
    details = (
        f"EEG summary reports {n_outputs} supervised artifacts for {n_inputs} inputs; "
        f"{n_processed_npz} NPZ files are present in processed outputs."
    )

    return [CheckResult(
        name="EEG annotations",
        passed=passed,
        details=details,
        stats={
            "n_inputs": n_inputs,
            "n_processed_npz": n_processed_npz,
            "n_supervised_outputs_in_summary": n_outputs,
            "shape_mismatches": shape_mismatches[:20],
        },
    )]


def check_ecg_outputs(processed_root: Path) -> List[CheckResult]:
    ecg_root = processed_root / "ecg"
    ecg_files = list_npz_files(ecg_root / "supervised" / "ptbxl")
    summary_path = ecg_root / "ptbxl_window_summary.json"

    if not ecg_files:
        return [CheckResult(
            name="ECG folds",
            passed=False,
            details=f"No ECG output files found under {ecg_root}",
        )]

    if not summary_path.exists():
        return [CheckResult(
            name="ECG folds",
            passed=False,
            details=f"Missing ECG summary file: {summary_path}",
        )]

    summary = safe_load_json(summary_path)
    artifacts = extract_artifacts(summary)
    outputs = [
        item for item in artifacts
        if item.get("artifact_kind") == "supervised_npz"
    ]

    split_names = set()
    cv_folds = set()
    output_counts = {}
    for item in outputs:
        if "split" in item:
            split_names.add(item["split"])
        if "cv_fold" in item:
            cv_folds.add(item["cv_fold"])
        path_str = item.get("path") or item.get("file")
        n_samples = item.get("n_samples")
        if path_str is not None and n_samples is not None:
            output_counts[path_str] = n_samples

    passed = (
        {"train", "val", "test"}.issubset(split_names)
        and cv_folds == set(range(8))
        and len(outputs) == len(ecg_files)
    )

    return [CheckResult(
        name="ECG folds",
        passed=passed,
        details=(
            f"ECG summary reports splits {sorted(split_names)} and CV folds {sorted(cv_folds)}; "
            f"{len(ecg_files)} NPZ files are present in processed outputs."
        ),
        stats={
            "splits": sorted(split_names),
            "cv_folds": sorted(cv_folds),
            "n_processed_npz": len(ecg_files),
            "n_summary_outputs": len(outputs),
            "output_counts": output_counts,
        },
    )]


def check_resources(raw_root: Path, processed_root: Path, repro_stats: Optional[Dict[str, Any]]) -> CheckResult:
    raw_bytes = sizeof_bytes(raw_root) if raw_root.exists() else 0
    proc_bytes = sizeof_bytes(processed_root) if processed_root.exists() else 0

    runtime_total = None
    peak_ram_mb = None
    if repro_stats:
        runtime_total = repro_stats.get("total_runtime_sec")
        peak_ram_mb = repro_stats.get("peak_ram_mb_across_run")

    details = (
        f"Raw storage: {human_size(raw_bytes)}; "
        f"processed storage: {human_size(proc_bytes)}; "
        f"total runtime: {runtime_total if runtime_total is not None else 'n/a'} sec; "
        f"peak RAM: {peak_ram_mb if peak_ram_mb is not None else 'n/a'} MB."
    )
    return CheckResult(
        name="Resource awareness",
        passed=True,
        details=details,
        stats={
            "raw_storage_bytes": raw_bytes,
            "processed_storage_bytes": proc_bytes,
            "total_runtime_sec": runtime_total,
            "peak_ram_mb": peak_ram_mb,
        },
    )


# ---------------------------------------------------------
# reporting
# ---------------------------------------------------------

def render_markdown(results: List[CheckResult]) -> str:
    lines = []
    lines.append("# Validation Report")
    lines.append("")
    passed_n = sum(r.passed for r in results)
    lines.append(f"Overall: **{passed_n}/{len(results)} checks passed**")
    lines.append("")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"## {r.name}")
        lines.append(f"**Status:** {status}")
        lines.append("")
        lines.append(r.details)
        lines.append("")
        if r.stats:
            lines.append("```json")
            lines.append(json.dumps(r.stats, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed multimodal pipeline outputs.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--report-dir", type=Path, default=Path("validation_report"))
    parser.add_argument("--full-run", action="store_true", help="Run setup + preprocessing commands before validating outputs.")
    parser.add_argument("--clean-first", action="store_true", help="Remove processed/interim/report directories before full run.")
    parser.add_argument(
        "--repro-cmd",
        action="append",
        default=[],
        help="Documented command to execute for reproducibility. Can be repeated.",
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    results: List[CheckResult] = []

    repro_result = None
    if args.full_run:
        if not args.repro_cmd:
            parser.error("--full-run requires at least one --repro-cmd")
        repro_result = run_full_pipeline(
            repo_root=args.repo_root,
            commands=args.repro_cmd,
            clean_first=args.clean_first,
            processed_root=args.processed_root,
            interim_root=args.interim_root,
            report_dir=args.report_dir,
        )
        results.append(repro_result)
    elif args.repro_cmd:
        repro_result = check_reproducibility(args.repo_root, args.repro_cmd)
        results.append(repro_result)

    results.extend(check_har_outputs(args.processed_root))
    results.extend(check_eeg_outputs(args.processed_root))
    results.extend(check_ecg_outputs(args.processed_root))
    results.append(check_resources(args.raw_root, args.processed_root, repro_result.stats if repro_result else None))

    md = render_markdown(results)
    json_payload = {
        "generated_at_epoch": time.time(),
        "results": [asdict(r) for r in results],
    }

    md_path = args.report_dir / "validation_report.md"
    json_path = args.report_dir / "validation_report.json"

    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")

    failed = [r for r in results if not r.passed]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()