#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
INTERIM_DIR="${PROJECT_ROOT}/data/interim"
PROCESSED_DIR="${PROJECT_ROOT}/data/processed"
MANIFEST_DIR="${PROJECT_ROOT}/data/manifests"
REPORTS_DIR="${PROJECT_ROOT}/reports"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_DIR="${PROJECT_ROOT}/configs"

mkdir -p \
  "${RAW_DIR}" \
  "${INTERIM_DIR}" \
  "${PROCESSED_DIR}" \
  "${MANIFEST_DIR}" \
  "${REPORTS_DIR}" \
  "${LOG_DIR}" \
  "${CONFIG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/setup_data_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] Starting setup at $(date)"
echo "[INFO] Project root: ${PROJECT_ROOT}"

download_file() {
  local url="$1"
  local out="$2"
  local tmp="${out}.part"

  # If final file exists, validate it before trusting it
  if [[ -f "${out}" ]]; then
    if unzip -tq "${out}" >/dev/null 2>&1; then
      echo "[INFO] Valid archive already exists, skipping download: ${out}"
      return 0
    else
      echo "[WARN] Existing file is invalid, removing: ${out}"
      rm -f "${out}"
    fi
  fi

  rm -f "${tmp}"

  echo "[INFO] Downloading: ${url}"
  curl -L --fail --retry 5 --retry-delay 5 --output "${tmp}" "${url}"

  # Validate before promoting to final filename
  if unzip -tq "${tmp}" >/dev/null 2>&1; then
    mv "${tmp}" "${out}"
    echo "[INFO] Download validated: ${out}"
  else
    echo "[ERROR] Downloaded file is not a valid ZIP: ${tmp}"
    file "${tmp}" || true
    rm -f "${tmp}"
    return 1
  fi
}

cleanup_extraction_markers() {
  local target_dir="$1"

  find "${target_dir}" -type f -name ".extract_done" -delete
}

extract_zip() {
  local zip_file="$1"
  local target_dir="$2"
  local marker="${target_dir}/.extract_done"

  if [[ -f "${marker}" ]]; then
    echo "[INFO] Extraction already completed, skipping: ${target_dir}"
    return 0
  fi

  if ! unzip -tq "${zip_file}" >/dev/null 2>&1; then
    echo "[ERROR] Invalid ZIP archive: ${zip_file}"
    file "${zip_file}" || true
    return 1
  fi

  rm -rf "${target_dir}"
  mkdir -p "${target_dir}"
  cleanup_extraction_markers "${target_dir}"

  echo "[INFO] Extracting ${zip_file} -> ${target_dir}"
  if unzip -o "${zip_file}" -d "${target_dir}" >/dev/null; then
    extract_nested_archives "${target_dir}"
    touch "${marker}"
  else
    echo "[ERROR] Extraction failed, cleaning partial output: ${target_dir}"
    rm -rf "${target_dir}"
    return 1
  fi
}

extract_nested_archives() {
  local target_dir="$1"
  local changed=1
  local pass=0

  while [[ ${changed} -eq 1 ]]; do
    changed=0
    pass=$((pass + 1))

    while IFS= read -r -d '' archive; do
      local archive_dir
      local archive_name
      local base_name
      local dest_dir
      local marker

      archive_dir="$(dirname "${archive}")"
      archive_name="$(basename "${archive}")"
      base_name="${archive_name%.*}"
      dest_dir="${archive_dir}/${base_name}"
      marker="${dest_dir}/.extract_done"

      if [[ -f "${marker}" ]]; then
        echo "[INFO] Nested archive already extracted, skipping: ${archive}"
        continue
      fi

      if unzip -tq "${archive}" >/dev/null 2>&1; then
        echo "[INFO] Extracting nested archive (pass ${pass}): ${archive} -> ${dest_dir}"
        mkdir -p "${dest_dir}"
        if unzip -o "${archive}" -d "${dest_dir}" >/dev/null; then
          touch "${marker}"
          changed=1
        else
          echo "[ERROR] Failed to extract nested archive: ${archive}"
          return 1
        fi
      fi
    done < <(find "${target_dir}" -type f -name "*.zip" -print0)
  done
}

write_sha256_manifest() {
  local target="$1"
  local manifest="$2"

  echo "[INFO] Writing SHA256 manifest for ${target}"
  find "${target}" -type f ! -name ".extract_done" -print0 | sort -z | xargs -0 sha256sum > "${manifest}"
}

# Dataset URLs
PAMAP_URL="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
WISDM_URL="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
MHEALTH_URL="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
EEGMMIDB_URL="https://physionet.org/content/eegmmidb/get-zip/1.0.0/"
PTBXL_URL="https://physionet.org/content/ptb-xl/get-zip/1.0.3/"

# Download archives
download_file "${PAMAP_URL}"   "${RAW_DIR}/pamap2.zip"
download_file "${WISDM_URL}"   "${RAW_DIR}/wisdm.zip"
download_file "${MHEALTH_URL}" "${RAW_DIR}/mhealth.zip"
download_file "${EEGMMIDB_URL}" "${RAW_DIR}/eegmmidb.zip"
download_file "${PTBXL_URL}"   "${RAW_DIR}/ptbxl.zip"

# Extract
extract_zip "${RAW_DIR}/pamap2.zip"   "${RAW_DIR}/pamap2"
extract_zip "${RAW_DIR}/wisdm.zip"    "${RAW_DIR}/wisdm"
extract_zip "${RAW_DIR}/mhealth.zip"  "${RAW_DIR}/mhealth"
extract_zip "${RAW_DIR}/eegmmidb.zip" "${RAW_DIR}/eegmmidb"
extract_zip "${RAW_DIR}/ptbxl.zip"    "${RAW_DIR}/ptbxl"

echo "[INFO] Verifying extracted dataset contents"
find "${RAW_DIR}/pamap2" -type f | head -20
find "${RAW_DIR}/wisdm" -type f | head -20
find "${RAW_DIR}/mhealth" -type f | head -20
find "${RAW_DIR}/eegmmidb" -type f | head -20
find "${RAW_DIR}/ptbxl" -type f | head -20

# Manifests
write_sha256_manifest "${RAW_DIR}/pamap2"   "${MANIFEST_DIR}/pamap2_sha256.txt"
write_sha256_manifest "${RAW_DIR}/wisdm"    "${MANIFEST_DIR}/wisdm_sha256.txt"
write_sha256_manifest "${RAW_DIR}/mhealth"  "${MANIFEST_DIR}/mhealth_sha256.txt"
write_sha256_manifest "${RAW_DIR}/eegmmidb" "${MANIFEST_DIR}/eegmmidb_sha256.txt"
write_sha256_manifest "${RAW_DIR}/ptbxl"    "${MANIFEST_DIR}/ptbxl_sha256.txt"

cat > "${MANIFEST_DIR}/download_manifest_${TIMESTAMP}.txt" <<EOF
timestamp=${TIMESTAMP}
project_root=${PROJECT_ROOT}
pamap_url=${PAMAP_URL}
wisdm_url=${WISDM_URL}
mhealth_url=${MHEALTH_URL}
eegmmidb_url=${EEGMMIDB_URL}
ptbxl_url=${PTBXL_URL}
EOF

echo "[INFO] Setup completed at $(date)"