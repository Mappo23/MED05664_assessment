#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path, PurePath

import pandas as pd
import yaml

HAR_DATASETS = ["pamap2", "wisdm"]

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


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_pamap2_subject_id(path: Path) -> str:
    m = re.search(r"subject(\d+)", path.stem.lower())
    return f"subject{m.group(1)}" if m else path.stem


def parse_pamap2_source_record_id(path: Path) -> str:
    return path.stem


def is_pamap2_protocol_file(path: Path) -> bool:
    return "protocol" in {part.lower() for part in path.parts}


def make_pamap2_output_name(path: Path) -> str:
    return f"{path.stem}.parquet"


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

    out = pd.DataFrame({
        "timestamp": merged["timestamp"],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--dataset", choices=["pamap2", "wisdm", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_root = Path(cfg["paths"]["raw_dir"])
    interim_root = Path(cfg["paths"]["interim_dir"]) / "har"
    ensure_dir(interim_root)

    summaries = []

    if args.dataset in ("pamap2", "all"):
        summaries.append(parse_all_pamap2(raw_root, interim_root))

    if args.dataset in ("wisdm", "all"):
        summaries.append(parse_all_wisdm(raw_root, interim_root))

    write_json(interim_root / "parse_summary.json", {"summaries": summaries})
    print(json.dumps({"status": "ok", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()