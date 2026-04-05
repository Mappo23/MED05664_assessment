#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import yaml
from scipy import signal

HAR_DATASETS = ["pamap2", "wisdm", "mhealth"]

MHEALTH_COLUMNS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_lead_1", "ecg_lead_2",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "arm_acc_x", "arm_acc_y", "arm_acc_z",
    "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
    "arm_mag_x", "arm_mag_y", "arm_mag_z",
    "activity_id",
]

MHEALTH_ACTIVITY_MAP = {
    0: "null",
    1: "standing",
    2: "sitting",
    3: "lying",
    4: "walking",
    5: "ascending_stairs",
    6: "waist_bends_forward",
    7: "frontal_arm_elevation",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back",
}

PAMAP2_COLUMNS = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temperature",
    "hand_acc_16g_x", "hand_acc_16g_y", "hand_acc_16g_z",
    "hand_acc_6g_x", "hand_acc_6g_y", "hand_acc_6g_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_ori_1", "hand_ori_2", "hand_ori_3", "hand_ori_4",
    "chest_temperature",
    "chest_acc_16g_x", "chest_acc_16g_y", "chest_acc_16g_z",
    "chest_acc_6g_x", "chest_acc_6g_y", "chest_acc_6g_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_ori_1", "chest_ori_2", "chest_ori_3", "chest_ori_4",
    "ankle_temperature",
    "ankle_acc_16g_x", "ankle_acc_16g_y", "ankle_acc_16g_z",
    "ankle_acc_6g_x", "ankle_acc_6g_y", "ankle_acc_6g_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_ori_1", "ankle_ori_2", "ankle_ori_3", "ankle_ori_4",
]

PAMAP2_ACTIVITY_MAP = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

HAR_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
DEFAULT_HAR_CFG = {
    "target_sampling_hz": 20,
    "cleaning": {
        "lowpass_hz": 8.0,
    },
    "windowing": {
        "pretrain": {
            "window_sec": 10.0,
            "overlap": 0.0,
            "include_labels": False,
        },
        "supervised": {
            "window_sec": 5.0,
            "overlap": 0.5,
            "include_labels": True,
            "min_label_purity": 0.8,
        },
    },
}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_npz(path: Path, **payload) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **payload)


def get_har_config(cfg: dict) -> dict:
    har_cfg = cfg.get("har", {}) if isinstance(cfg, dict) else {}
    merged = {
        "target_sampling_hz": har_cfg.get("target_sampling_hz", DEFAULT_HAR_CFG["target_sampling_hz"]),
        "cleaning": {
            "lowpass_hz": har_cfg.get("cleaning", {}).get(
                "lowpass_hz", DEFAULT_HAR_CFG["cleaning"]["lowpass_hz"]
            ),
        },
        "windowing": {
            "pretrain": {
                "window_sec": har_cfg.get("windowing", {}).get("pretrain", {}).get(
                    "window_sec", DEFAULT_HAR_CFG["windowing"]["pretrain"]["window_sec"]
                ),
                "overlap": har_cfg.get("windowing", {}).get("pretrain", {}).get(
                    "overlap", DEFAULT_HAR_CFG["windowing"]["pretrain"]["overlap"]
                ),
                "include_labels": False,
            },
            "supervised": {
                "window_sec": har_cfg.get("windowing", {}).get("supervised", {}).get(
                    "window_sec", DEFAULT_HAR_CFG["windowing"]["supervised"]["window_sec"]
                ),
                "overlap": har_cfg.get("windowing", {}).get("supervised", {}).get(
                    "overlap", DEFAULT_HAR_CFG["windowing"]["supervised"]["overlap"]
                ),
                "include_labels": True,
                "min_label_purity": har_cfg.get("windowing", {}).get("supervised", {}).get(
                    "min_label_purity", DEFAULT_HAR_CFG["windowing"]["supervised"]["min_label_purity"]
                ),
            },
        },
    }
    return merged


def infer_time_scale_seconds(timestamp_series: pd.Series) -> float:
    ts = pd.to_numeric(timestamp_series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if ts.size < 2:
        return 1.0

    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0

    median_dt = float(np.median(diffs))
    if median_dt > 1e8:
        return 1e9
    if median_dt > 1e5:
        return 1e6
    if median_dt > 1e2:
        return 1e3
    return 1.0


def build_relative_time_seconds(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    scale = infer_time_scale_seconds(df["timestamp"])
    ts = ts / scale
    valid = np.isfinite(ts)
    if not valid.any():
        return np.arange(len(df), dtype=np.float64)
    first_valid = ts[valid][0]
    ts = ts - first_valid
    return ts

def normalize_discontinuous_time_series(
    timestamp_series: pd.Series,
    max_gap_factor: float = 5.0,
) -> np.ndarray:
    ts = pd.to_numeric(timestamp_series, errors="coerce").to_numpy(dtype=np.float64)
    if ts.size == 0:
        return ts

    valid = np.isfinite(ts)
    if not valid.any():
        return np.arange(len(ts), dtype=np.float64)

    ts = ts.copy()
    first_valid = ts[valid][0]
    ts[valid] = ts[valid] - first_valid

    if ts.size == 1:
        ts[0] = 0.0
        return ts

    diffs = np.diff(ts)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive_diffs.size == 0:
        return np.arange(len(ts), dtype=np.float64)

    nominal_dt = float(np.median(positive_diffs))
    if not np.isfinite(nominal_dt) or nominal_dt <= 0:
        return np.arange(len(ts), dtype=np.float64)

    max_allowed_gap = nominal_dt * max_gap_factor
    adjusted_diffs = diffs.copy()

    invalid_mask = ~np.isfinite(adjusted_diffs) | (adjusted_diffs <= 0)
    adjusted_diffs[invalid_mask] = nominal_dt

    large_gap_mask = adjusted_diffs > max_allowed_gap
    adjusted_diffs[large_gap_mask] = nominal_dt

    normalized = np.empty_like(ts)
    normalized[0] = 0.0
    normalized[1:] = np.cumsum(adjusted_diffs)
    return normalized

def estimate_sampling_rate_hz(time_sec: np.ndarray) -> float:
    diffs = np.diff(time_sec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.nan
    return float(1.0 / np.median(diffs))


def maybe_lowpass_filter(values: np.ndarray, source_hz: float, cutoff_hz: float) -> np.ndarray:
    if not np.isfinite(source_hz) or source_hz <= 0:
        return values
    if values.shape[0] < 16:
        return values
    nyquist = 0.5 * source_hz
    if cutoff_hz >= nyquist:
        return values

    b, a = signal.butter(4, cutoff_hz / nyquist, btype="low")
    try:
        return signal.filtfilt(b, a, values, axis=0)
    except ValueError:
        return values


def interpolate_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[columns] = out[columns].interpolate(method="linear", limit_direction="both")
    return out


def robust_channel_standardize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    median = np.nanmedian(values, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=0, keepdims=True)
    return ((values - median) / (1.4826 * mad + eps)).astype(np.float32)


def resample_labels_nearest(time_old: np.ndarray, labels: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    if labels.size == 0:
        return np.array([], dtype=object)

    idx = np.searchsorted(time_old, time_new, side="left")
    idx = np.clip(idx, 0, len(time_old) - 1)

    left_idx = np.clip(idx - 1, 0, len(time_old) - 1)
    right_idx = idx
    choose_left = np.abs(time_new - time_old[left_idx]) <= np.abs(time_old[right_idx] - time_new)
    nearest_idx = np.where(choose_left, left_idx, right_idx)
    return labels[nearest_idx]


def clean_and_resample_frame(df: pd.DataFrame, target_hz: int, lowpass_hz: float) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work = work.sort_values(["subject_id", "timestamp", "row_idx"]).reset_index(drop=True)
    work = interpolate_numeric_frame(work, HAR_CHANNELS)

    time_sec = build_relative_time_seconds(work)
    source_hz = estimate_sampling_rate_hz(time_sec)

    values = work[HAR_CHANNELS].to_numpy(dtype=np.float64)
    values = maybe_lowpass_filter(values, source_hz=source_hz, cutoff_hz=lowpass_hz)

    if len(time_sec) < 2:
        raise ValueError("Not enough samples for resampling")

    duration = time_sec[-1]
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration for resampling")

    step = 1.0 / target_hz
    time_new = np.arange(0.0, duration + 1e-9, step, dtype=np.float64)
    if time_new.size < 2:
        raise ValueError("Resampled timeline too short")

    resampled = np.column_stack([
        np.interp(time_new, time_sec, values[:, i]) for i in range(values.shape[1])
    ])
    resampled = robust_channel_standardize(resampled)

    label_names = resample_labels_nearest(time_sec, work["raw_label_name"].astype(object).to_numpy(), time_new)
    raw_labels = resample_labels_nearest(time_sec, work["raw_label"].astype(object).to_numpy(), time_new)

    out = pd.DataFrame({
        "time_sec": time_new,
        "dataset": work["dataset"].iloc[0],
        "subject_id": work["subject_id"].iloc[0],
        "source_file": work["source_file"].iloc[0],
        "source_record_id": work["source_record_id"].iloc[0],
        "acc_x": resampled[:, 0],
        "acc_y": resampled[:, 1],
        "acc_z": resampled[:, 2],
        "gyro_x": resampled[:, 3],
        "gyro_y": resampled[:, 4],
        "gyro_z": resampled[:, 5],
        "raw_label": list(raw_labels),
        "raw_label_name": list(label_names),
    })
    out["row_idx"] = range(len(out))

    stats = {
        "source_hz_estimated": None if not np.isfinite(source_hz) else float(source_hz),
        "target_hz": int(target_hz),
        "n_input_rows": int(len(df)),
        "n_output_rows": int(len(out)),
        "nan_fraction_before_interp": float(df[HAR_CHANNELS].isna().mean().mean()),
    }
    return out, stats


def make_cleaned_output_name(path: Path) -> str:
    return f"{path.stem}.parquet"


def compute_window_label(labels: np.ndarray, min_label_purity: float) -> tuple[object, float] | tuple[None, float]:
    labels = pd.Series(labels, dtype="object")
    labels = labels.dropna()
    labels = labels[labels.astype(str).str.strip() != ""]
    if labels.empty:
        return None, 0.0

    counts = labels.value_counts()
    label = counts.index[0]
    purity = float(counts.iloc[0] / counts.sum())
    if purity < min_label_purity:
        return None, purity
    return label, purity


def build_windows_from_cleaned_frame(
    df: pd.DataFrame,
    window_sec: float,
    overlap: float,
    include_labels: bool,
    sampling_hz: int,
    min_label_purity: float,
) -> tuple[np.ndarray, list[dict]]:
    window_size = int(round(window_sec * sampling_hz))
    stride = int(round(window_size * (1.0 - overlap)))
    if stride <= 0:
        raise ValueError("Window stride must be positive")

    values = df[HAR_CHANNELS].to_numpy(dtype=np.float32)
    labels = df["raw_label_name"].astype(object).to_numpy()
    meta = []
    windows = []

    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window = values[start:end]
        if window.shape[0] != window_size:
            continue

        label_value = None
        label_purity = None
        if include_labels:
            label_value, label_purity = compute_window_label(labels[start:end], min_label_purity=min_label_purity)
            if label_value is None:
                continue

        windows.append(window)
        meta.append({
            "dataset": df["dataset"].iloc[0],
            "subject_id": df["subject_id"].iloc[0],
            "source_file": df["source_file"].iloc[0],
            "source_record_id": df["source_record_id"].iloc[0],
            "start_idx": int(start),
            "end_idx": int(end),
            "start_time_sec": float(df["time_sec"].iloc[start]),
            "end_time_sec": float(df["time_sec"].iloc[end - 1]),
            "sampling_hz": int(sampling_hz),
            "window_sec": float(window_sec),
            "label_name": None if label_value is None else str(label_value),
            "label_purity": None if label_purity is None else float(label_purity),
        })

    if not windows:
        return np.empty((0, window_size, len(HAR_CHANNELS)), dtype=np.float32), meta
    return np.stack(windows).astype(np.float32), meta


def discover_wisdm_sensor_files(wisdm_root: Path) -> dict:
    """
    Try to find watch accelerometer and watch gyroscope files.
    WISDM packaging can vary, so use a defensive search.
    """
    all_files = [p for p in wisdm_root.rglob("*") if p.is_file()]

    watch_accel = []
    watch_gyro = []

    for p in all_files:
        name = p.name.lower()
        suffix = p.suffix.lower()

        if suffix != ".txt":
            continue
        if name.startswith("."):
            continue
        if "watch" not in name:
            continue

        if "accel" in name:
            watch_accel.append(p)
        elif "gyro" in name or "gyroscope" in name:
            watch_gyro.append(p)

    return {
        "watch_accel": sorted(watch_accel),
        "watch_gyro": sorted(watch_gyro),
    }


def read_wisdm_table(path: Path) -> pd.DataFrame:
    """
    WISDM watch raw files are comma-separated text files with rows like:
    subject,activity,timestamp,x,y,z;
    Some files can contain malformed trailing lines, so read defensively.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["subject_id", "raw_label_name", "timestamp", "x", "y", "z"],
        dtype=str,
        sep=",",
        engine="python",
        on_bad_lines="skip",
    )

    df = df.apply(lambda col: col.map(lambda x: x.strip().rstrip(";") if isinstance(x, str) else x))
    df = df.dropna(how="all")
    return df


def normalize_wisdm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected raw WISDM row shape is commonly:
    subject, activity, timestamp, x, y, z
    but we keep this defensive.
    """
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected WISDM shape: {df.shape}")

    df = df.iloc[:, :6].copy()
    df.columns = ["subject_id", "raw_label_name", "timestamp", "x", "y", "z"]

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["raw_label_name"] = df["raw_label_name"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    df = df.dropna(subset=["subject_id", "raw_label_name", "timestamp", "x", "y", "z"])
    df = df[df["subject_id"] != ""]
    df = df.reset_index(drop=True)
    return df


def parse_wisdm_subject_token(path: Path) -> str:
    m = re.search(r"data_(\d+)", path.stem.lower())
    return m.group(1) if m else path.stem


def pair_wisdm_watch_streams(accel_path: Path, gyro_path: Path) -> pd.DataFrame:
    acc = normalize_wisdm_columns(read_wisdm_table(accel_path))
    gyr = normalize_wisdm_columns(read_wisdm_table(gyro_path))

    merged = pd.merge(
        acc,
        gyr,
        on=["subject_id", "raw_label_name", "timestamp"],
        how="inner",
        suffixes=("_acc", "_gyro"),
    )

    merged = merged.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

    timestamp_sec = pd.to_numeric(merged["timestamp"], errors="coerce").astype("float64") / 1e9
    timestamp_sec = pd.Series(
        normalize_discontinuous_time_series(timestamp_sec, max_gap_factor=5.0),
        index=merged.index,
    )

    out = pd.DataFrame({
        "timestamp": timestamp_sec,
        "dataset": "wisdm",
        "subject_id": merged["subject_id"],
        "source_file": f"{accel_path}|{gyro_path}",
        "source_record_id": parse_wisdm_subject_token(accel_path),
        "acc_x": merged["x_acc"],
        "acc_y": merged["y_acc"],
        "acc_z": merged["z_acc"],
        "gyro_x": merged["x_gyro"],
        "gyro_y": merged["y_gyro"],
        "gyro_z": merged["z_gyro"],
        "raw_label": pd.NA,
        "raw_label_name": merged["raw_label_name"],
    })

    out = out.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
    out["row_idx"] = range(len(out))
    return out

def parse_all_pamap2(raw_root: Path, out_root: Path) -> dict:
    pamap_root = raw_root / "pamap2"
    parsed_root = out_root / "pamap2" / "parsed"
    ensure_dir(parsed_root)

    all_dat_files = sorted(pamap_root.rglob("*.dat"))
    dat_files = [path for path in all_dat_files if is_pamap2_protocol_file(path)]
    summary = {
        "dataset": "pamap2",
        "selection": "protocol_only",
        "n_files_discovered": len(all_dat_files),
        "n_files": 0,
        "n_rows_total": 0,
        "outputs": [],
    }

    for path in dat_files:
        df = parse_pamap2_file(path)

        rel_name = make_pamap2_output_name(path)
        out_path = parsed_root / rel_name
        df.to_parquet(out_path, index=False)

        summary["n_files"] += 1
        summary["n_rows_total"] += len(df)
        summary["outputs"].append(str(out_path))

    write_json(parsed_root / "summary.json", summary)
    return summary


def build_wisdm_pairs(sensor_files: dict) -> list[tuple[Path, Path]]:
    """
    Pair accel and gyro files by filename stem after removing obvious sensor tokens.
    """
    def normalize_stem(p: Path) -> str:
        s = p.stem.lower()
        s = s.replace("accel", "").replace("gyro", "").replace("gyroscope", "")
        s = s.replace("watch", "")
        s = s.replace("__", "_").strip("_")
        return s

    gyro_map = {}
    for g in sensor_files["watch_gyro"]:
        gyro_map[normalize_stem(g)] = g

    pairs = []
    for a in sensor_files["watch_accel"]:
        key = normalize_stem(a)
        if key in gyro_map:
            pairs.append((a, gyro_map[key]))

    return pairs


def parse_all_wisdm(raw_root: Path, out_root: Path) -> dict:
    wisdm_root = raw_root / "wisdm"
    parsed_root = out_root / "wisdm" / "parsed"
    ensure_dir(parsed_root)

    sensor_files = discover_wisdm_sensor_files(wisdm_root)
    pairs = build_wisdm_pairs(sensor_files)

    summary = {
        "dataset": "wisdm",
        "selection": "watch_txt_only",
        "n_watch_accel_files": len(sensor_files["watch_accel"]),
        "n_watch_gyro_files": len(sensor_files["watch_gyro"]),
        "n_pairs": len(pairs),
        "n_outputs": 0,
        "n_rows_total": 0,
        "outputs": [],
    }

    for accel_path, gyro_path in pairs:
        try:
            df = pair_wisdm_watch_streams(accel_path, gyro_path)
        except Exception as e:
            print(f"[WARN] Failed pairing {accel_path.name} with {gyro_path.name}: {e}")
            continue

        if df.empty:
            print(f"[WARN] Empty merged WISDM pair: {accel_path.name} vs {gyro_path.name}")
            continue

        out_path = parsed_root / f"{PurePath(accel_path).stem}.parquet"
        df.to_parquet(out_path, index=False)

        summary["n_outputs"] += 1
        summary["n_rows_total"] += len(df)
        summary["outputs"].append(str(out_path))

    write_json(parsed_root / "summary.json", summary)
    return summary


def clean_all_har_dataset(dataset: str, cfg: dict, interim_root: Path) -> dict:
    har_cfg = get_har_config(cfg)
    parsed_root = interim_root / dataset / "parsed"
    cleaned_root = interim_root / dataset / "cleaned"
    ensure_dir(cleaned_root)

    parsed_files = sorted(parsed_root.glob("*.parquet"))
    summary = {
        "dataset": dataset,
        "stage": "clean_resample",
        "target_hz": har_cfg["target_sampling_hz"],
        "channel_schema": HAR_CHANNELS,
        "n_inputs": len(parsed_files),
        "n_outputs": 0,
        "outputs": [],
        "records": [],
    }

    for path in parsed_files:
        df = pd.read_parquet(path)
        try:
            cleaned_df, stats = clean_and_resample_frame(
                df,
                target_hz=har_cfg["target_sampling_hz"],
                lowpass_hz=har_cfg["cleaning"]["lowpass_hz"],
            )
        except Exception as e:
            print(f"[WARN] Failed cleaning {path.name}: {e}")
            continue

        out_path = cleaned_root / make_cleaned_output_name(path)
        cleaned_df.to_parquet(out_path, index=False)

        summary["n_outputs"] += 1
        summary["outputs"].append(str(out_path))
        summary["records"].append({"file": str(path), **stats})

    write_json(cleaned_root / "summary.json", summary)
    return summary


def window_all_har_dataset(dataset: str, cfg: dict, interim_root: Path, processed_root: Path) -> dict:
    har_cfg = get_har_config(cfg)
    cleaned_root = interim_root / dataset / "cleaned"
    cleaned_files = sorted(cleaned_root.glob("*.parquet"))

    pretrain_root = processed_root / "har" / "pretrain" / dataset
    supervised_root = processed_root / "har" / "supervised" / dataset
    ensure_dir(pretrain_root)
    ensure_dir(supervised_root)

    summary = {
        "dataset": dataset,
        "stage": "window",
        "channel_schema": HAR_CHANNELS,
        "n_inputs": len(cleaned_files),
        "pretrain_outputs": [],
        "supervised_outputs": [],
    }

    for path in cleaned_files:
        df = pd.read_parquet(path)

        X_pre, meta_pre = build_windows_from_cleaned_frame(
            df,
            window_sec=har_cfg["windowing"]["pretrain"]["window_sec"],
            overlap=har_cfg["windowing"]["pretrain"]["overlap"],
            include_labels=False,
            sampling_hz=har_cfg["target_sampling_hz"],
            min_label_purity=0.0,
        )
        pre_out = pretrain_root / f"{path.stem}.npz"
        write_npz(
            pre_out,
            X=X_pre,
            channels=np.array(HAR_CHANNELS, dtype=object),
            metadata=np.array(meta_pre, dtype=object),
        )
        summary["pretrain_outputs"].append({"file": str(pre_out), "n_windows": int(X_pre.shape[0])})

        X_sup, meta_sup = build_windows_from_cleaned_frame(
            df,
            window_sec=har_cfg["windowing"]["supervised"]["window_sec"],
            overlap=har_cfg["windowing"]["supervised"]["overlap"],
            include_labels=True,
            sampling_hz=har_cfg["target_sampling_hz"],
            min_label_purity=har_cfg["windowing"]["supervised"]["min_label_purity"],
        )
        sup_out = supervised_root / f"{path.stem}.npz"
        write_npz(
            sup_out,
            X=X_sup,
            channels=np.array(HAR_CHANNELS, dtype=object),
            metadata=np.array(meta_sup, dtype=object),
        )
        summary["supervised_outputs"].append({"file": str(sup_out), "n_windows": int(X_sup.shape[0])})

    write_json(processed_root / "har" / f"{dataset}_window_summary.json", summary)
    return summary


def run_har_pipeline_for_dataset(dataset: str, cfg: dict, raw_root: Path, interim_root: Path, processed_root: Path) -> list[dict]:
    summaries = []
    if dataset == "pamap2":
        summaries.append(parse_all_pamap2(raw_root, interim_root))
    elif dataset == "wisdm":
        summaries.append(parse_all_wisdm(raw_root, interim_root))
    elif dataset == "mhealth":
        summaries.append(parse_all_mhealth(raw_root, interim_root))
    else:
        raise ValueError(f"Unsupported HAR dataset: {dataset}")

    summaries.append(clean_all_har_dataset(dataset, cfg, interim_root))
    summaries.append(window_all_har_dataset(dataset, cfg, interim_root, processed_root))
    return summaries

def parse_mhealth_subject_id(path: Path) -> str:
    m = re.search(r"subject(\d+)", path.stem.lower())
    return f"subject{m.group(1)}" if m else path.stem


def parse_mhealth_source_record_id(path: Path) -> str:
    return path.stem


def make_mhealth_output_name(path: Path) -> str:
    return f"{path.stem}.parquet"

def parse_pamap2_subject_id(path: Path) -> str:
    m = re.search(r"subject(\d+)", path.stem.lower())
    return f"subject{m.group(1)}" if m else path.stem


def parse_pamap2_source_record_id(path: Path) -> str:
    return path.stem


def is_pamap2_protocol_file(path: Path) -> bool:
    return "protocol" in {part.lower() for part in path.parts}


def make_pamap2_output_name(path: Path) -> str:
    return f"{path.stem}.parquet"


def parse_mhealth_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=MHEALTH_COLUMNS,
        na_values=["NaN"],
        engine="python",
    )

    timestamp = np.arange(len(df), dtype=np.float64) / 50.0

    out = pd.DataFrame({
        "timestamp": timestamp,
        "dataset": "mhealth",
        "subject_id": parse_mhealth_subject_id(path),
        "source_file": str(path),
        "source_record_id": parse_mhealth_source_record_id(path),
        # use right lower arm IMU as common wrist/watch proxy
        "acc_x": pd.to_numeric(df["arm_acc_x"], errors="coerce"),
        "acc_y": pd.to_numeric(df["arm_acc_y"], errors="coerce"),
        "acc_z": pd.to_numeric(df["arm_acc_z"], errors="coerce"),
        "gyro_x": pd.to_numeric(df["arm_gyro_x"], errors="coerce"),
        "gyro_y": pd.to_numeric(df["arm_gyro_y"], errors="coerce"),
        "gyro_z": pd.to_numeric(df["arm_gyro_z"], errors="coerce"),
        "raw_label": pd.to_numeric(df["activity_id"], errors="coerce").astype("Int64"),
    })

    out["raw_label_name"] = out["raw_label"].map(MHEALTH_ACTIVITY_MAP)
    out["row_idx"] = range(len(out))
    return out


def parse_all_mhealth(raw_root: Path, out_root: Path) -> dict:
    mhealth_root = raw_root / "mhealth"
    parsed_root = out_root / "mhealth" / "parsed"
    ensure_dir(parsed_root)

    log_files = sorted(mhealth_root.rglob("*.log"))
    summary = {
        "dataset": "mhealth",
        "selection": "subject_logs_only",
        "source_sampling_hz": 50,
        "shared_sensor_proxy": "right_lower_arm",
        "null_label_mapping": {
            "raw_label": 0,
            "mapped_label_name": "null",
            "meaning": "background / non-target class kept explicitly documented in parsed outputs",
        },
        "n_files": 0,
        "n_rows_total": 0,
        "outputs": [],
    }

    for path in log_files:
        df = parse_mhealth_file(path)

        out_path = parsed_root / make_mhealth_output_name(path)
        df.to_parquet(out_path, index=False)

        summary["n_files"] += 1
        summary["n_rows_total"] += len(df)
        summary["outputs"].append(str(out_path))

    write_json(parsed_root / "summary.json", summary)
    return summary


def parse_pamap2_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        na_values=["NaN"],
        engine="python",
    )

    out = pd.DataFrame({
        "timestamp": pd.to_numeric(df["timestamp"], errors="coerce"),
        "dataset": "pamap2",
        "subject_id": parse_pamap2_subject_id(path),
        "source_file": str(path),
        "source_record_id": parse_pamap2_source_record_id(path),
        # use wrist/hand IMU as common watch/wrist proxy
        "acc_x": pd.to_numeric(df["hand_acc_16g_x"], errors="coerce"),
        "acc_y": pd.to_numeric(df["hand_acc_16g_y"], errors="coerce"),
        "acc_z": pd.to_numeric(df["hand_acc_16g_z"], errors="coerce"),
        "gyro_x": pd.to_numeric(df["hand_gyro_x"], errors="coerce"),
        "gyro_y": pd.to_numeric(df["hand_gyro_y"], errors="coerce"),
        "gyro_z": pd.to_numeric(df["hand_gyro_z"], errors="coerce"),
        "raw_label": pd.to_numeric(df["activity_id"], errors="coerce").astype("Int64"),
    })

    out["raw_label_name"] = out["raw_label"].map(PAMAP2_ACTIVITY_MAP)
    out["row_idx"] = range(len(out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--dataset", choices=["pamap2", "wisdm", "mhealth", "all"], default="all")
    parser.add_argument("--stage", choices=["parse", "clean", "window", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_root = Path(cfg["paths"]["raw_dir"])
    interim_root = Path(cfg["paths"]["interim_dir"]) / "har"
    processed_root = Path(cfg["paths"]["processed_dir"])
    ensure_dir(interim_root)
    ensure_dir(processed_root / "har")

    summaries = []
    datasets = HAR_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        if args.stage == "parse":
            if dataset == "pamap2":
                summaries.append(parse_all_pamap2(raw_root, interim_root))
            elif dataset == "wisdm":
                summaries.append(parse_all_wisdm(raw_root, interim_root))
            elif dataset == "mhealth":
                summaries.append(parse_all_mhealth(raw_root, interim_root))
        elif args.stage == "clean":
            summaries.append(clean_all_har_dataset(dataset, cfg, interim_root))
        elif args.stage == "window":
            summaries.append(window_all_har_dataset(dataset, cfg, interim_root, processed_root))
        else:
            summaries.extend(run_har_pipeline_for_dataset(dataset, cfg, raw_root, interim_root, processed_root))

    write_json(interim_root / "pipeline_summary.json", {"summaries": summaries})
    print(json.dumps({"status": "ok", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()