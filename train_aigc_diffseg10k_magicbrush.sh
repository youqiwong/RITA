#!/usr/bin/env bash
set -euo pipefail

# Official RITA training settings are retained in train.py:
# 512x512, 512 epochs, per-source sample count 1840, AdamW 1e-4,
# cosine schedule, and the official three-step autoregressive objective.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/train/RITA_AIGC_DiffSeg10k_MagicBrush}"
DIFFSEG_TXT="${DIFFSEG_TXT:-/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/DiffSeg30k/DiffSeg30k.txt}"
MAGICBRUSH_TXT="${MAGICBRUSH_TXT:-/home/tione/notebook/users/youqiwang/Agent_All/Datasets/ori/train/MagicBrush/MagicBrush_train.txt}"
MIT_B3_PATH="${MIT_B3_PATH:-/home/tione/notebook/users/youqiwang/Mesorch/pretrained/mit_b3.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/runs/rita_official_diffseg10k_magicbrush}"
BATCH_SIZE="${BATCH_SIZE:-8}"

[[ -f "${DIFFSEG_TXT}" ]] || { echo "Missing DiffSeg manifest: ${DIFFSEG_TXT}" >&2; exit 2; }
[[ -f "${MAGICBRUSH_TXT}" ]] || { echo "Missing MagicBrush manifest: ${MAGICBRUSH_TXT}" >&2; exit 2; }
[[ -f "${MIT_B3_PATH}" ]] || { echo "Missing MiT-B3 checkpoint: ${MIT_B3_PATH}" >&2; exit 2; }
mkdir -p "${DATA_ROOT}"
if [[ ! -s "${DATA_ROOT}/rita_aigc_config.json" || \
      ! -s "${DATA_ROOT}/rita_aigc_val_config.json" || \
      ! -s "${DATA_ROOT}/DiffSeg30k_first10000_train.json" ]]; then
  python "${REPO_DIR}/scripts/prepare_aigc_diffseg_magicbrush.py" \
    --diffseg-txt "${DIFFSEG_TXT}" \
    --magicbrush-txt "${MAGICBRUSH_TXT}" \
    --output-dir "${DATA_ROOT}" \
    --diffseg-limit 10000 \
    --check-files
fi

python "${REPO_DIR}/scripts/audit_aigc_pairs.py" \
  --config "${DATA_ROOT}/rita_aigc_config.json"

export RITA_MIT_B3_PATH="${MIT_B3_PATH}"
mkdir -p "${OUTPUT_DIR}/ckpts" "${OUTPUT_DIR}/images" "${OUTPUT_DIR}/runs"
cd "${REPO_DIR}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train.py \
  --path "${OUTPUT_DIR}" \
  --image_size 512 \
  --epoch 512 \
  --batch_size "${BATCH_SIZE}" \
  --data_path "${DATA_ROOT}/rita_aigc_config.json" \
  --val_config "${DATA_ROOT}/rita_aigc_val_config.json"
