# MED05664 Assessment — Reproducible Pipeline

Minimal commands to reproduce preprocessing and validation.

---

## 1. Clone repository

```bash
git clone https://github.com/Mappo23/MED05664_assessment.git
cd MED05664_assessment
```

## 2. Setup data

```bash
bash setup_data.sh
```

## 3. Create reproducible environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## HAR preprocessing

### PAMAP2

```bash
python3 preprocess_har.py --dataset pamap2 --stage parse
python3 preprocess_har.py --dataset pamap2 --stage clean
python3 preprocess_har.py --dataset pamap2 --stage window
```

### WISDM

```bash
python3 preprocess_har.py --dataset wisdm --stage parse
python3 preprocess_har.py --dataset wisdm --stage clean
python3 preprocess_har.py --dataset wisdm --stage window
```

---

## EEG preprocessing (EEGMMIDB)

```bash
python3 preprocess_eeg.py --dataset eegmmidb --stage parse
python3 preprocess_eeg.py --dataset eegmmidb --stage clean
python3 preprocess_eeg.py --dataset eegmmidb --stage window
```

---

## ECG preprocessing (PTB-XL)

```bash
python3 preprocess_ecg.py --dataset ptbxl --stage parse
python3 preprocess_ecg.py --dataset ptbxl --stage clean
python3 preprocess_ecg.py --dataset ptbxl --stage window
```

---

## Validation (optional)

```bash
python3 validate_pipeline.py \
  --repo-root . \
  --raw-root data/raw \
  --processed-root data/processed \
  --interim-root data/interim \
  --report-dir validation_report
```

---

## Outputs

Processed data:

```
data/processed/
├── har/
│   ├── pretrain/
│   └── supervised/
├── eeg/
└── ecg/
```

Validation report:

```
validation_report/
├── validation_report.md
└── validation_report.json
```

---

## Requirements

Python ≥ 3.9

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Notes

- HAR target sampling: 20 Hz  
- EEG target sampling: 160 Hz  
- ECG sampling: 100 Hz  
- HAR pretrain windows: 10s  
- HAR supervised windows: 5s (50% overlap)  
