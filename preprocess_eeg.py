#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

EEG_DATASETS = ["eegmmidb"]

EEGMMIDB_DEFAULT_RUNS = [4, 8, 12]
EEG_CHANNELS_DEFAULT = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
]

DEFAULT_EEG_CFG = {
    "runs": [4, 8, 12],
    "target_sampling_hz": 160,
    "window_sec": 4.0,
    "event_codes": ["T1", "T2"],
    "channel_selection": EEG_CHANNELS_DEFAULT,
    "cleaning": {
        "rereference": "average",
        "bandpass_hz": [8.0, 30.0],
        "notch_hz": 50.0,
        "normalize": "zscore_channel",
        "amplitude_threshold_uv": 200.0,
        "flat_threshold_uv": 0.5,
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


def get_eeg_config(cfg: dict) -> dict:
    eeg_cfg = cfg.get("eeg", {}) if isinstance(cfg, dict) else {}
    merged = {
        "runs": eeg_cfg.get("runs", DEFAULT_EEG_CFG["runs"]),
        "target_sampling_hz": eeg_cfg.get(
            "target_sampling_hz", DEFAULT_EEG_CFG["target_sampling_hz"]
        ),
        "window_sec": eeg_cfg.get("window_sec", DEFAULT_EEG_CFG["window_sec"]),
        "event_codes": eeg_cfg.get("event_codes", DEFAULT_EEG_CFG["event_codes"]),
        "channel_selection": eeg_cfg.get(
            "channel_selection", DEFAULT_EEG_CFG["channel_selection"]
        ),
        "cleaning": {
            "rereference": eeg_cfg.get("cleaning", {}).get(
                "rereference", DEFAULT_EEG_CFG["cleaning"]["rereference"]
            ),
            "bandpass_hz": eeg_cfg.get("cleaning", {}).get(
                "bandpass_hz", DEFAULT_EEG_CFG["cleaning"]["bandpass_hz"]
            ),
            "notch_hz": eeg_cfg.get("cleaning", {}).get(
                "notch_hz", DEFAULT_EEG_CFG["cleaning"]["notch_hz"]
            ),
            "normalize": eeg_cfg.get("cleaning", {}).get(
                "normalize", DEFAULT_EEG_CFG["cleaning"]["normalize"]
            ),
            "amplitude_threshold_uv": eeg_cfg.get("cleaning", {}).get(
                "amplitude_threshold_uv",
                DEFAULT_EEG_CFG["cleaning"]["amplitude_threshold_uv"],
            ),
            "flat_threshold_uv": eeg_cfg.get("cleaning", {}).get(
                "flat_threshold_uv", DEFAULT_EEG_CFG["cleaning"]["flat_threshold_uv"]
            ),
        },
    }
    return merged


def parse_eegmmidb_subject_id(path: Path) -> str:
    m = re.search(r"(S\d{3})", path.as_posix(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else path.parent.name.upper()


def parse_eegmmidb_run_id(path: Path) -> int:
    m = re.search(r"R(\d{2})", path.stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse run id from {path}")
    return int(m.group(1))


def parse_eegmmidb_source_record_id(path: Path) -> str:
    return path.stem


def make_eegmmidb_output_name(path: Path) -> str:
    return f"{path.stem}.parquet"


def discover_eegmmidb_edf_files(raw_root: Path, selected_runs: list[int]) -> list[Path]:
    eeg_root = raw_root / "eegmmidb"
    all_edf_files = sorted(eeg_root.rglob("*.edf"))
    selected = []

    for path in all_edf_files:
        try:
            run_id = parse_eegmmidb_run_id(path)
        except ValueError:
            continue
        if run_id in selected_runs:
            selected.append(path)

    return selected


def map_event_code_to_label(event_code: str) -> str | None:
    if event_code == "T0":
        return "rest"
    if event_code == "T1":
        return "left_imagery"
    if event_code == "T2":
        return "right_imagery"
    return None


def normalize_channel_names(raw: mne.io.BaseRaw) -> None:
    rename_map = {}
    for ch in raw.ch_names:
        normalized = ch.strip(".").capitalize()
        rename_map[ch] = normalized
    raw.rename_channels(rename_map)


def parse_eegmmidb_file(path: Path, eeg_cfg: dict) -> pd.DataFrame:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    normalize_channel_names(raw)

    subject_id = parse_eegmmidb_subject_id(path)
    run_id = parse_eegmmidb_run_id(path)
    source_record_id = parse_eegmmidb_source_record_id(path)
    sfreq = float(raw.info["sfreq"])

    rows = []
    for ann_idx, ann in enumerate(raw.annotations):
        event_code = str(ann["description"]).strip()
        if event_code not in eeg_cfg["event_codes"]:
            continue

        label_name = map_event_code_to_label(event_code)
        if label_name is None:
            continue

        rows.append(
            {
                "dataset": "eegmmidb",
                "subject_id": subject_id,
                "run_id": run_id,
                "source_file": str(path),
                "source_record_id": source_record_id,
                "annotation_idx": ann_idx,
                "event_code": event_code,
                "label_name": label_name,
                "onset_sec": float(ann["onset"]),
                "duration_sec": float(ann["duration"]),
                "sampling_hz": sfreq,
                "n_channels_total": int(len(raw.ch_names)),
                "channels_total": json.dumps(list(raw.ch_names)),
            }
        )

    return pd.DataFrame(rows)


def parse_all_eegmmidb(raw_root: Path, out_root: Path, cfg: dict) -> dict:
    eeg_cfg = get_eeg_config(cfg)
    parsed_root = out_root / "eegmmidb" / "parsed"
    ensure_dir(parsed_root)

    edf_files = discover_eegmmidb_edf_files(raw_root, eeg_cfg["runs"])
    summary = {
        "dataset": "eegmmidb",
        "selection": "motor_imagery_runs_4_8_12",
        "runs": eeg_cfg["runs"],
        "event_codes": eeg_cfg["event_codes"],
        "n_files": len(edf_files),
        "n_events_total": 0,
        "outputs": [],
    }

    for path in edf_files:
        try:
            df = parse_eegmmidb_file(path, eeg_cfg)
        except Exception as e:
            print(f"[WARN] Failed parsing {path.name}: {e}")
            continue

        out_path = parsed_root / make_eegmmidb_output_name(path)
        df.to_parquet(out_path, index=False)

        summary["n_events_total"] += len(df)
        summary["outputs"].append(str(out_path))

    write_json(parsed_root / "summary.json", summary)
    return summary


def select_available_channels(raw: mne.io.BaseRaw, requested_channels: list[str]) -> list[str]:
    available = set(raw.ch_names)
    selected = [ch for ch in requested_channels if ch in available]
    if not selected:
        raise ValueError("No requested EEG channels found in record")
    return selected


def maybe_rereference(raw: mne.io.BaseRaw, rereference: str) -> mne.io.BaseRaw:
    if rereference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif rereference == "none":
        pass
    else:
        raise ValueError(f"Unsupported rereference: {rereference}")
    return raw


def maybe_bandpass_filter(raw: mne.io.BaseRaw, bandpass_hz: list[float] | None) -> mne.io.BaseRaw:
    if not bandpass_hz:
        return raw
    raw.filter(
        l_freq=float(bandpass_hz[0]),
        h_freq=float(bandpass_hz[1]),
        verbose="ERROR",
    )
    return raw


def maybe_notch_filter(raw: mne.io.BaseRaw, notch_hz: float | None) -> mne.io.BaseRaw:
    if notch_hz is None:
        return raw
    raw.notch_filter(freqs=[float(notch_hz)], verbose="ERROR")
    return raw


def maybe_resample_raw(raw: mne.io.BaseRaw, target_hz: int) -> mne.io.BaseRaw:
    current_hz = float(raw.info["sfreq"])
    if int(round(current_hz)) == int(target_hz):
        return raw
    raw.resample(target_hz, verbose="ERROR")
    return raw


def normalize_window(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values.astype(np.float32)

    if mode == "zscore_channel":
        mean = np.mean(values, axis=1, keepdims=True)
        std = np.std(values, axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return ((values - mean) / std).astype(np.float32)

    raise ValueError(f"Unsupported normalize mode: {mode}")


def reject_window(
    values: np.ndarray,
    amplitude_threshold_uv: float,
    flat_threshold_uv: float,
) -> str | None:
    values_uv = values * 1e6

    if not np.isfinite(values_uv).all():
        return "non_finite"

    peak_abs_uv = float(np.max(np.abs(values_uv)))
    if peak_abs_uv > amplitude_threshold_uv:
        return f"amp_gt_{amplitude_threshold_uv:.1f}uV"

    channel_std_uv = np.std(values_uv, axis=1)
    if np.any(channel_std_uv < flat_threshold_uv):
        return f"flat_lt_{flat_threshold_uv:.1f}uV"

    return None


def clean_and_epoch_eeg_record(
    edf_path: Path,
    event_df: pd.DataFrame,
    eeg_cfg: dict,
) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    normalize_channel_names(raw)
    eeg_channels = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])
    raw.pick(eeg_channels)

    selected_channels = select_available_channels(raw, eeg_cfg["channel_selection"])
    raw.pick(selected_channels)

    raw = maybe_rereference(raw, eeg_cfg["cleaning"]["rereference"])
    raw = maybe_bandpass_filter(raw, eeg_cfg["cleaning"]["bandpass_hz"])
    raw = maybe_notch_filter(raw, eeg_cfg["cleaning"]["notch_hz"])
    raw = maybe_resample_raw(raw, eeg_cfg["target_sampling_hz"])

    sampling_hz = int(round(raw.info["sfreq"]))
    window_size = int(round(eeg_cfg["window_sec"] * sampling_hz))

    kept_windows = []
    kept_meta = []
    rejected_meta = []

    for _, row in event_df.iterrows():
        onset_sec = float(row["onset_sec"])
        start_idx = int(round(onset_sec * sampling_hz))
        end_idx = start_idx + window_size

        if end_idx > raw.n_times:
            rejected_meta.append(
                {
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "run_id": int(row["run_id"]),
                    "source_file": row["source_file"],
                    "source_record_id": row["source_record_id"],
                    "annotation_idx": int(row["annotation_idx"]),
                    "event_code": row["event_code"],
                    "label_name": row["label_name"],
                    "onset_sec": onset_sec,
                    "reason": "short_tail",
                }
            )
            continue

        values = raw.get_data(start=start_idx, stop=end_idx)
        reason = reject_window(
            values,
            amplitude_threshold_uv=eeg_cfg["cleaning"]["amplitude_threshold_uv"],
            flat_threshold_uv=eeg_cfg["cleaning"]["flat_threshold_uv"],
        )
        if reason is not None:
            rejected_meta.append(
                {
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "run_id": int(row["run_id"]),
                    "source_file": row["source_file"],
                    "source_record_id": row["source_record_id"],
                    "annotation_idx": int(row["annotation_idx"]),
                    "event_code": row["event_code"],
                    "label_name": row["label_name"],
                    "onset_sec": onset_sec,
                    "reason": reason,
                }
            )
            continue

        values = normalize_window(values, eeg_cfg["cleaning"]["normalize"])
        kept_windows.append(values)
        kept_meta.append(
            {
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "run_id": int(row["run_id"]),
                "source_file": row["source_file"],
                "source_record_id": row["source_record_id"],
                "annotation_idx": int(row["annotation_idx"]),
                "event_code": row["event_code"],
                "label_name": row["label_name"],
                "onset_sec": onset_sec,
                "window_start_sec": onset_sec,
                "duration_sec": float(eeg_cfg["window_sec"]),
                "window_end_sec": onset_sec + float(eeg_cfg["window_sec"]),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "sampling_hz": int(sampling_hz),
                "n_channels": int(values.shape[0]),
                "n_samples": int(values.shape[1]),
                "channels": json.dumps(selected_channels),
            }
        )

    return kept_windows, kept_meta, rejected_meta


def clean_all_eeg_dataset(dataset: str, cfg: dict, raw_root: Path, interim_root: Path) -> dict:
    if dataset != "eegmmidb":
        raise ValueError(f"Unsupported EEG dataset: {dataset}")

    eeg_cfg = get_eeg_config(cfg)
    parsed_root = interim_root / dataset / "parsed"
    cleaned_root = interim_root / dataset / "cleaned"
    ensure_dir(cleaned_root)

    parsed_files = sorted(parsed_root.glob("*.parquet"))
    summary = {
        "dataset": dataset,
        "stage": "clean_event_preprocess",
        "target_hz": eeg_cfg["target_sampling_hz"],
        "window_sec": eeg_cfg["window_sec"],
        "n_inputs": len(parsed_files),
        "n_outputs": 0,
        "n_windows_total": 0,
        "n_rejected_total": 0,
        "outputs": [],
        "records": [],
    }

    for path in parsed_files:
        event_df = pd.read_parquet(path)
        if event_df.empty:
            continue

        edf_path = Path(event_df["source_file"].iloc[0])

        try:
            windows, kept_meta, rejected_meta = clean_and_epoch_eeg_record(
                edf_path=edf_path,
                event_df=event_df,
                eeg_cfg=eeg_cfg,
            )
        except Exception as e:
            print(f"[WARN] Failed cleaning {path.name}: {e}")
            continue

        stem = path.stem
        meta_out = cleaned_root / f"{stem}.parquet"
        rej_out = cleaned_root / f"{stem}_rejections.parquet"
        npy_out = cleaned_root / f"{stem}.npy"

        if kept_meta:
            np.save(npy_out, np.stack(windows).astype(np.float32))
            pd.DataFrame(kept_meta).to_parquet(meta_out, index=False)
            summary["n_outputs"] += 1
            summary["n_windows_total"] += len(kept_meta)
            summary["outputs"].append(str(meta_out))
        else:
            pd.DataFrame(columns=[
                "dataset", "subject_id", "run_id", "source_file", "source_record_id",
                "annotation_idx", "event_code", "label_name", "onset_sec",
                "duration_sec", "start_idx", "end_idx", "sampling_hz",
                "n_channels", "n_samples", "channels"
            ]).to_parquet(meta_out, index=False)

        pd.DataFrame(rejected_meta).to_parquet(rej_out, index=False)
        summary["n_rejected_total"] += len(rejected_meta)
        summary["records"].append(
            {
                "file": str(path),
                "source_edf": str(edf_path),
                "n_events_in": int(len(event_df)),
                "n_windows_out": int(len(kept_meta)),
                "n_rejected": int(len(rejected_meta)),
            }
        )

    write_json(cleaned_root / "summary.json", summary)
    return summary


def build_windows_from_cleaned_eeg_record(
    array_path: Path,
    meta_path: Path,
) -> tuple[np.ndarray, list[dict]]:
    X = np.load(array_path)
    meta_df = pd.read_parquet(meta_path)
    metadata = meta_df.to_dict(orient="records")

    if X.ndim != 3:
        raise ValueError(f"Expected 3D EEG array, got shape {X.shape}")

    return X.astype(np.float32), metadata


def window_all_eeg_dataset(dataset: str, cfg: dict, interim_root: Path, processed_root: Path) -> dict:
    if dataset != "eegmmidb":
        raise ValueError(f"Unsupported EEG dataset: {dataset}")

    cleaned_root = interim_root / dataset / "cleaned"
    cleaned_meta_files = sorted(
        [p for p in cleaned_root.glob("*.parquet") if not p.name.endswith("_rejections.parquet")]
    )

    supervised_root = processed_root / "eeg" / "supervised" / dataset
    ensure_dir(supervised_root)

    summary = {
        "dataset": dataset,
        "stage": "window",
        "n_inputs": len(cleaned_meta_files),
        "supervised_outputs": [],
    }

    for meta_path in cleaned_meta_files:
        array_path = cleaned_root / f"{meta_path.stem}.npy"
        if not array_path.exists():
            continue

        try:
            X, metadata = build_windows_from_cleaned_eeg_record(array_path, meta_path)
        except Exception as e:
            print(f"[WARN] Failed window packaging {meta_path.name}: {e}")
            continue

        out_path = supervised_root / f"{meta_path.stem}.npz"
        channel_names = []
        if metadata:
            try:
                channel_names = json.loads(metadata[0]["channels"])
            except Exception:
                channel_names = []

        write_npz(
            out_path,
            X=X,
            channels=np.array(channel_names, dtype=object),
            metadata=np.array(metadata, dtype=object),
        )
        summary["supervised_outputs"].append(
            {"file": str(out_path), "n_windows": int(X.shape[0])}
        )

    write_json(processed_root / "eeg" / f"{dataset}_window_summary.json", summary)
    return summary


def run_eeg_pipeline_for_dataset(
    dataset: str,
    cfg: dict,
    raw_root: Path,
    interim_root: Path,
    processed_root: Path,
) -> list[dict]:
    summaries = []
    if dataset == "eegmmidb":
        summaries.append(parse_all_eegmmidb(raw_root, interim_root, cfg))
    else:
        raise ValueError(f"Unsupported EEG dataset: {dataset}")

    summaries.append(clean_all_eeg_dataset(dataset, cfg, raw_root, interim_root))
    summaries.append(window_all_eeg_dataset(dataset, cfg, interim_root, processed_root))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eeg_harmonization.yaml")
    parser.add_argument("--dataset", choices=["eegmmidb", "all"], default="all")
    parser.add_argument("--stage", choices=["parse", "clean", "window", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if "paths" not in cfg:
        raise ValueError(
            f"Missing 'paths' section in config: {args.config}\n"
            "Expected keys: paths.raw_dir, paths.interim_dir, paths.processed_dir"
        )

    raw_root = Path(cfg["paths"]["raw_dir"])
    interim_root = Path(cfg["paths"]["interim_dir"]) / "eeg"
    processed_root = Path(cfg["paths"]["processed_dir"])
    ensure_dir(interim_root)
    ensure_dir(processed_root / "eeg")

    summaries = []
    datasets = EEG_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        if args.stage == "parse":
            if dataset == "eegmmidb":
                summaries.append(parse_all_eegmmidb(raw_root, interim_root, cfg))
        elif args.stage == "clean":
            summaries.append(clean_all_eeg_dataset(dataset, cfg, raw_root, interim_root))
        elif args.stage == "window":
            summaries.append(window_all_eeg_dataset(dataset, cfg, interim_root, processed_root))
        else:
            summaries.extend(
                run_eeg_pipeline_for_dataset(dataset, cfg, raw_root, interim_root, processed_root)
            )

    write_json(interim_root / "pipeline_summary.json", {"summaries": summaries})
    print(json.dumps({"status": "ok", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()