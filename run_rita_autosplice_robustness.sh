#!/usr/bin/env bash
set -euo pipefail

RITA=/pubdata/wangyq/Projects/RITA
DATA=/pubdata/wangyq/Projects/Datasets/AIGC-Loc-Testsets
PY=/pubdata/wangyq/anaconda3/envs/qwen_vl_env/bin/python
CKPT="${CKPT:-${DATA}/RITA_DiffSeg10k_MagicBrush/runs/rita_official_diffseg10k_magicbrush/ckpts/best.pth}"
OUT="${OUT:-${RITA}/robustness_results/autosplice_n1000_seed42}"
MANIFEST="${OUT}/manifests/autosplice_n1000_seed42.txt"

export PYTHONNOUSERSITE=1
export RITA_MIT_B3_PATH="${RITA}/pretrained/mit_b3.pth"
cd "${RITA}"

read -ra GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"
if [ "${#GPU_LIST[@]}" -ne 6 ]; then
  echo "RITA AutoSplice robustness requires exactly 6 GPUs; got: ${GPU_LIST[*]}" >&2
  exit 2
fi

"${PY}" prepare_rita_autosplice_robustness.py \
  --output-dir "${OUT}/manifests" --sample-count 1000 --seed 42 --workers 32

PIDS=()
for rank in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" "${PY}" infer_rita_autosplice_robustness.py \
    --ckpt "${CKPT}" --manifest "${MANIFEST}" --out-dir "${OUT}/pred_masks" \
    --num-chunks 6 --chunk-idx "${rank}" --seed 42 &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [ "${FAIL}" -ne 0 ]; then
  echo "At least one RITA robustness shard failed." >&2
  exit 1
fi

"${PY}" eval_rita_autosplice_robustness.py \
  --manifest "${MANIFEST}" --pred-root "${OUT}/pred_masks" \
  --output-dir "${OUT}/tsv" --method RITA-Retrain

echo "Long TSV: ${OUT}/tsv/autosplice_robustness_long.tsv"
echo "F1 TSV: ${OUT}/tsv/autosplice_robustness_f1.tsv"
echo "IoU TSV: ${OUT}/tsv/autosplice_robustness_iou.tsv"
