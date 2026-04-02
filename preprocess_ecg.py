#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from scipy import signal

ECG_DATASETS = ["ptbxl"]

PTBXL_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

DEFAULT_ECG_CFG = {
    "sampling_rate_hz": 100,  # PTB-XL supports 100 or 500
    "cleaning": {
        "highpass_hz": 0.5,
        "lowpass_hz": 45.0,
        "normalize": "zscore_lead",
    },
    "splits": {
        "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
        "val_folds": [9],
        "test_folds": [10],
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


def get_ecg_config(cfg: dict) -> dict:
    ecg_cfg = cfg.get("ecg", {}) if isinstance(cfg, dict) else {}

    sampling_rate_hz = int(ecg_cfg.get("sampling_rate_hz", DEFAULT_ECG_CFG["sampling_rate_hz"]))
    if sampling_rate_hz not in (100, 500):
        raise ValueError("ecg.sampling_rate_hz must be either 100 or 500 for PTB-XL")

    merged = {
        "sampling_rate_hz": sampling_rate_hz,
        "cleaning": {
            "highpass_hz": ecg_cfg.get("cleaning", {}).get(
                "highpass_hz", DEFAULT_ECG_CFG["cleaning"]["highpass_hz"]
            ),
            "lowpass_hz": ecg_cfg.get("cleaning", {}).get(
                "lowpass_hz", DEFAULT_ECG_CFG["cleaning"]["lowpass_hz"]
            ),
            "normalize": ecg_cfg.get("cleaning", {}).get(
                "normalize", DEFAULT_ECG_CFG["cleaning"]["normalize"]
            ),
        },
        "splits": {
            "train_folds": ecg_cfg.get("splits", {}).get(
                "train_folds", DEFAULT_ECG_CFG["splits"]["train_folds"]
            ),
            "val_folds": ecg_cfg.get("splits", {}).get(
                "val_folds", DEFAULT_ECG_CFG["splits"]["val_folds"]
            ),
            "test_folds": ecg_cfg.get("splits", {}).get(
                "test_folds", DEFAULT_ECG_CFG["splits"]["test_folds"]
            ),
        },
    }
    return merged


def parse_scp_codes(value) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def assign_split(strat_fold: int, split_cfg: dict) -> str:
    if strat_fold in split_cfg["train_folds"]:
        return "train"
    if strat_fold in split_cfg["val_folds"]:
        return "val"
    if strat_fold in split_cfg["test_folds"]:
        return "test"
    return "unused"


def assign_cv_fold(strat_fold: int, split_cfg: dict) -> int | None:
    train_folds = list(split_cfg["train_folds"])
    if strat_fold not in train_folds:
        return None
    return int(train_folds.index(strat_fold))


# Helper to robustly find the PTB-XL dataset root directory
def find_ptbxl_base_dir(raw_root: Path) -> Path:
    ptb_root = raw_root / "ptbxl"
    direct_db = ptb_root / "ptbxl_database.csv"
    if direct_db.exists():
        return ptb_root

    subdir_matches = [
        p.parent
        for p in ptb_root.glob("*/ptbxl_database.csv")
        if p.is_file()
    ]
    if len(subdir_matches) == 1:
        return subdir_matches[0]
    if len(subdir_matches) > 1:
        raise FileNotFoundError(
            f"Multiple PTB-XL dataset roots found under {ptb_root}: {subdir_matches}"
        )

    raise FileNotFoundError(
        f"Missing PTB-XL metadata file under {ptb_root} (checked direct path and one subfolder level)"
    )


def load_ptbxl_metadata(raw_root: Path, ecg_cfg: dict) -> pd.DataFrame:
    ptb_root = raw_root / "ptbxl"
    dataset_root = find_ptbxl_base_dir(raw_root)
    db_path = dataset_root / "ptbxl_database.csv"

    df = pd.read_csv(db_path)

    df["scp_codes_dict"] = df["scp_codes"].apply(parse_scp_codes)
    df["label_codes"] = df["scp_codes_dict"].apply(lambda d: sorted(list(d.keys())))
    df["label_scores"] = df["scp_codes_dict"].apply(lambda d: json.dumps(d, sort_keys=True))

    filename_col = "filename_lr" if int(ecg_cfg["sampling_rate_hz"]) == 100 else "filename_hr"
    if filename_col not in df.columns:
        raise ValueError(f"Missing expected PTB-XL column: {filename_col}")

    out = pd.DataFrame({
        "dataset": "ptbxl",
        "patient_id": df["patient_id"].astype("Int64"),
        "record_id": df["ecg_id"].astype("Int64"),
        "source_file": df[filename_col].astype(str),
        "sampling_rate_hz": int(ecg_cfg["sampling_rate_hz"]),
        "lead_names": [json.dumps(PTBXL_LEADS)] * len(df),
        "labels": df["label_codes"].apply(json.dumps),
        "label_scores": df["label_scores"],
        "strat_fold": df["strat_fold"].astype("Int64"),
        "split": df["strat_fold"].apply(lambda x: assign_split(int(x), ecg_cfg["splits"])),
        "cv_fold": df["strat_fold"].apply(lambda x: assign_cv_fold(int(x), ecg_cfg["splits"])),
    })

    out = out[out["split"] != "unused"].reset_index(drop=True)
    return out


def parse_all_ptbxl(raw_root: Path, out_root: Path, cfg: dict) -> dict:
    ecg_cfg = get_ecg_config(cfg)
    parsed_root = out_root / "ptbxl" / "parsed"
    ensure_dir(parsed_root)

    meta_df = load_ptbxl_metadata(raw_root, ecg_cfg)
    out_path = parsed_root / "ptbxl_records.parquet"
    meta_df.to_parquet(out_path, index=False)

    summary = {
        "dataset": "ptbxl",
        "stage": "parse",
        "sampling_rate_hz": int(ecg_cfg["sampling_rate_hz"]),
        "n_records": int(len(meta_df)),
        "n_patients": int(meta_df["patient_id"].nunique()),
        "n_train": int((meta_df["split"] == "train").sum()),
        "n_val": int((meta_df["split"] == "val").sum()),
        "n_test": int((meta_df["split"] == "test").sum()),
        "output": str(out_path),
    }

    write_json(parsed_root / "summary.json", summary)
    return summary


def maybe_bandpass_filter(values: np.ndarray, fs: int, highpass_hz: float | None, lowpass_hz: float | None) -> np.ndarray:
    if highpass_hz is None and lowpass_hz is None:
        return values

    nyquist = fs / 2.0
    if values.shape[0] < max(16, fs // 2):
        return values

    if highpass_hz is not None and lowpass_hz is not None:
        wn = [float(highpass_hz) / nyquist, float(lowpass_hz) / nyquist]
        if wn[0] <= 0 or wn[1] >= 1 or wn[0] >= wn[1]:
            return values
        btype = "band"
    elif highpass_hz is not None:
        wn = float(highpass_hz) / nyquist
        if wn <= 0 or wn >= 1:
            return values
        btype = "high"
    else:
        wn = float(lowpass_hz) / nyquist
        if wn <= 0 or wn >= 1:
            return values
        btype = "low"

    b, a = signal.butter(4, wn, btype=btype)
    try:
        return signal.filtfilt(b, a, values, axis=0)
    except Exception:
        return values


def normalize_ecg(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values.astype(np.float32)

    if mode == "zscore_lead":
        mean = np.mean(values, axis=0, keepdims=True)
        std = np.std(values, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return ((values - mean) / std).astype(np.float32)

    raise ValueError(f"Unsupported ECG normalize mode: {mode}")


def load_ptbxl_waveform(raw_root: Path, relative_record_path: str) -> tuple[np.ndarray, list[str]]:
    dataset_root = find_ptbxl_base_dir(raw_root)
    record_path = dataset_root / relative_record_path
    signal_data, meta = wfdb.rdsamp(str(record_path))
    lead_names = list(meta.get("sig_name", []))
    return signal_data.astype(np.float32), lead_names


def clean_ptbxl_record(
    raw_root: Path,
    row: pd.Series,
    ecg_cfg: dict,
) -> tuple[np.ndarray, dict]:
    values, lead_names = load_ptbxl_waveform(raw_root, row["source_file"])
    fs = int(row["sampling_rate_hz"])

    if values.ndim != 2:
        raise ValueError(f"Expected 2D ECG waveform, got shape {values.shape}")
    if values.shape[1] != 12:
        raise ValueError(f"Expected 12 leads, got shape {values.shape}")

    values = maybe_bandpass_filter(
        values,
        fs=fs,
        highpass_hz=ecg_cfg["cleaning"]["highpass_hz"],
        lowpass_hz=ecg_cfg["cleaning"]["lowpass_hz"],
    )
    values = normalize_ecg(values, ecg_cfg["cleaning"]["normalize"])

    meta = {
        "dataset": row["dataset"],
        "patient_id": int(row["patient_id"]),
        "record_id": int(row["record_id"]),
        "source_file": row["source_file"],
        "sampling_rate_hz": int(fs),
        "lead_names": json.dumps(lead_names),
        "labels": row["labels"],
        "label_scores": row["label_scores"],
        "strat_fold": int(row["strat_fold"]),
        "split": row["split"],
        "cv_fold": None if pd.isna(row["cv_fold"]) else int(row["cv_fold"]),
        "n_samples": int(values.shape[0]),
        "n_leads": int(values.shape[1]),
        "duration_sec": float(values.shape[0] / fs),
    }
    return values, meta


def clean_all_ecg_dataset(dataset: str, cfg: dict, raw_root: Path, interim_root: Path) -> dict:
    if dataset != "ptbxl":
        raise ValueError(f"Unsupported ECG dataset: {dataset}")

    ecg_cfg = get_ecg_config(cfg)
    parsed_root = interim_root / dataset / "parsed"
    cleaned_root = interim_root / dataset / "cleaned"
    ensure_dir(cleaned_root)

    meta_path = parsed_root / "ptbxl_records.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing parsed PTB-XL metadata: {meta_path}")

    parsed_df = pd.read_parquet(meta_path)

    kept_meta = []
    arrays = []
    failures = []

    for _, row in parsed_df.iterrows():
        try:
            values, meta = clean_ptbxl_record(raw_root, row, ecg_cfg)
            arrays.append(values)
            kept_meta.append(meta)
        except Exception as e:
            failures.append(
                {
                    "record_id": int(row["record_id"]),
                    "patient_id": int(row["patient_id"]),
                    "source_file": row["source_file"],
                    "reason": str(e),
                }
            )

    if not arrays:
        raise RuntimeError("No PTB-XL ECG records could be cleaned")

    X = np.stack(arrays).astype(np.float32)
    cleaned_meta_df = pd.DataFrame(kept_meta)

    npy_out = cleaned_root / "ptbxl_waveforms.npy"
    meta_out = cleaned_root / "ptbxl_waveforms.parquet"
    rej_out = cleaned_root / "ptbxl_rejections.parquet"

    np.save(npy_out, X)
    cleaned_meta_df.to_parquet(meta_out, index=False)
    pd.DataFrame(failures).to_parquet(rej_out, index=False)

    summary = {
        "dataset": dataset,
        "stage": "clean",
        "sampling_rate_hz": int(ecg_cfg["sampling_rate_hz"]),
        "n_records_in": int(len(parsed_df)),
        "n_records_out": int(len(cleaned_meta_df)),
        "n_rejected": int(len(failures)),
        "waveform_shape": [int(x) for x in X.shape],
        "outputs": [str(npy_out), str(meta_out), str(rej_out)],
    }

    write_json(cleaned_root / "summary.json", summary)
    return summary


def build_split_package(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    split_name: str,
) -> tuple[np.ndarray, list[dict]]:
    sub_df = meta_df[meta_df["split"] == split_name].reset_index(drop=True)
    if sub_df.empty:
        return np.empty((0,) + X.shape[1:], dtype=np.float32), []

    idx = sub_df.index.to_numpy()
    X_sub = X[idx].astype(np.float32)
    meta = sub_df.to_dict(orient="records")
    return X_sub, meta


def build_cv_package(
    X: np.ndarray,
    meta_df: pd.DataFrame,
    cv_fold: int,
) -> tuple[np.ndarray, list[dict]]:
    sub_df = meta_df[(meta_df["split"] == "train") & (meta_df["cv_fold"] == cv_fold)].reset_index(drop=True)
    if sub_df.empty:
        return np.empty((0,) + X.shape[1:], dtype=np.float32), []

    idx = sub_df.index.to_numpy()
    X_sub = X[idx].astype(np.float32)
    meta = sub_df.to_dict(orient="records")
    return X_sub, meta


def window_all_ecg_dataset(dataset: str, cfg: dict, interim_root: Path, processed_root: Path) -> dict:
    if dataset != "ptbxl":
        raise ValueError(f"Unsupported ECG dataset: {dataset}")

    ecg_cfg = get_ecg_config(cfg)
    cleaned_root = interim_root / dataset / "cleaned"
    x_path = cleaned_root / "ptbxl_waveforms.npy"
    meta_path = cleaned_root / "ptbxl_waveforms.parquet"

    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Missing cleaned PTB-XL outputs required for packaging")

    X = np.load(x_path)
    meta_df = pd.read_parquet(meta_path).reset_index(drop=True)

    supervised_root = processed_root / "ecg" / "supervised" / dataset
    ensure_dir(supervised_root)

    summary = {
        "dataset": dataset,
        "stage": "window",
        "sampling_rate_hz": int(ecg_cfg["sampling_rate_hz"]),
        "outputs": [],
    }

    for split_name in ["train", "val", "test"]:
        mask = meta_df["split"] == split_name
        sub_df = meta_df[mask].reset_index(drop=True)
        if sub_df.empty:
            continue

        original_idx = np.where(mask.to_numpy())[0]
        X_sub = X[original_idx].astype(np.float32)
        out_path = supervised_root / f"{split_name}.npz"

        write_npz(
            out_path,
            X=X_sub,
            leads=np.array(PTBXL_LEADS, dtype=object),
            metadata=np.array(sub_df.to_dict(orient="records"), dtype=object),
        )
        summary["outputs"].append(
            {"file": str(out_path), "split": split_name, "n_records": int(X_sub.shape[0])}
        )

    for cv_fold in range(len(ecg_cfg["splits"]["train_folds"])):
        mask = (meta_df["split"] == "train") & (meta_df["cv_fold"] == cv_fold)
        sub_df = meta_df[mask].reset_index(drop=True)
        if sub_df.empty:
            continue

        original_idx = np.where(mask.to_numpy())[0]
        X_sub = X[original_idx].astype(np.float32)
        out_path = supervised_root / f"train_cv_fold_{cv_fold}.npz"

        write_npz(
            out_path,
            X=X_sub,
            leads=np.array(PTBXL_LEADS, dtype=object),
            metadata=np.array(sub_df.to_dict(orient="records"), dtype=object),
        )
        summary["outputs"].append(
            {"file": str(out_path), "split": "train", "cv_fold": int(cv_fold), "n_records": int(X_sub.shape[0])}
        )

    write_json(processed_root / "ecg" / f"{dataset}_window_summary.json", summary)
    return summary


def run_ecg_pipeline_for_dataset(
    dataset: str,
    cfg: dict,
    raw_root: Path,
    interim_root: Path,
    processed_root: Path,
) -> list[dict]:
    summaries = []
    if dataset == "ptbxl":
        summaries.append(parse_all_ptbxl(raw_root, interim_root, cfg))
    else:
        raise ValueError(f"Unsupported ECG dataset: {dataset}")

    summaries.append(clean_all_ecg_dataset(dataset, cfg, raw_root, interim_root))
    summaries.append(window_all_ecg_dataset(dataset, cfg, interim_root, processed_root))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ecg_harmonization.yaml")
    parser.add_argument("--dataset", choices=["ptbxl", "all"], default="all")
    parser.add_argument("--stage", choices=["parse", "clean", "window", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if "paths" not in cfg:
        raise ValueError(
            f"Missing 'paths' section in config: {args.config}\n"
            "Expected keys: paths.raw_dir, paths.interim_dir, paths.processed_dir"
        )

    raw_root = Path(cfg["paths"]["raw_dir"])
    interim_root = Path(cfg["paths"]["interim_dir"]) / "ecg"
    processed_root = Path(cfg["paths"]["processed_dir"])
    ensure_dir(interim_root)
    ensure_dir(processed_root / "ecg")

    summaries = []
    datasets = ECG_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        if args.stage == "parse":
            if dataset == "ptbxl":
                summaries.append(parse_all_ptbxl(raw_root, interim_root, cfg))
        elif args.stage == "clean":
            summaries.append(clean_all_ecg_dataset(dataset, cfg, raw_root, interim_root))
        elif args.stage == "window":
            summaries.append(window_all_ecg_dataset(dataset, cfg, interim_root, processed_root))
        else:
            summaries.extend(
                run_ecg_pipeline_for_dataset(dataset, cfg, raw_root, interim_root, processed_root)
            )

    write_json(interim_root / "pipeline_summary.json", {"summaries": summaries})
    print(json.dumps({"status": "ok", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()