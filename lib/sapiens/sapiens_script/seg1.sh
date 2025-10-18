#!/bin/bash
set -euo pipefail

cd ../../.. || exit

# -----------------------------------------------
# CONFIGURATION (adjust only paths/checkpoint names if needed)
# -----------------------------------------------
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE='torchscript'   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"

SPLIT="train"
IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/${SPLIT}/img"

# Outputs
SEG_OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/seg"
DEPTH_OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/depth"

MODEL_NAME='sapiens_1b'

# --- Checkpoints & runners
# SEG
SEG_CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/checkpoints/${MODEL_NAME}/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_${MODE}.pt2"
SEG_RUN_FILE="/home/clark/Documents/GitHub/sapiens/lite/demo/vis_seg1.py"

# DEPTH  (update filename if yours differs)
DEPTH_CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/depth/checkpoints/${MODEL_NAME}/sapiens_1b_goliath_best_${MODE}.pt2"
DEPTH_RUN_FILE="/home/clark/Documents/GitHub/sapiens/lite/demo/vis_depth1.py"

# Common
INPUT_LIST="${IMG_ROOT}/image_list_views_0to4.txt"
SEG_OUTPUT_DIR="${SEG_OUTPUT_ROOT}/${MODEL_NAME}"
DEPTH_OUTPUT_DIR="${DEPTH_OUTPUT_ROOT}/${MODEL_NAME}"

BATCH_SIZE=16
GPU_ID=0
HEIGHT=1024
WIDTH=768

# -----------------------------------------------
# BUILD FLAT IMAGE LIST (views 0..4)
# -----------------------------------------------
echo "[seg_then_depth.sh] Scanning for views 0..4 (jpg/png) under ${IMG_ROOT}"
find "${IMG_ROOT}" -type f \( \
  -name "0.jpg" -o -name "1.jpg" -o -name "2.jpg" -o -name "3.jpg" -o -name "4.jpg" -o \
  -name "0.png" -o -name "1.png" -o -name "2.png" -o -name "3.png" -o -name "4.png" \
\) | sort > "${INPUT_LIST}"

if [ ! -s "${INPUT_LIST}" ]; then
  echo "[seg_then_depth.sh] No images found for 0..4. Check IMG_ROOT=${IMG_ROOT}"
  exit 1
fi

echo "[seg_then_depth.sh] Found $(wc -l < "${INPUT_LIST}") images (views 0..4)"
echo "[seg_then_depth.sh] SEG out:   ${SEG_OUTPUT_DIR}"
echo "[seg_then_depth.sh] DEPTH out: ${DEPTH_OUTPUT_DIR}"
mkdir -p "${SEG_OUTPUT_DIR}" "${DEPTH_OUTPUT_DIR}"

# -----------------------------------------------
# STEP 1: SEGMENTATION (views 0..4)
# -----------------------------------------------
echo "[seg_then_depth.sh] === SEGMENTATION ==="
echo "  checkpoint: ${SEG_CHECKPOINT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${SEG_RUN_FILE}" \
  "${SEG_CHECKPOINT}" \
  --input "${INPUT_LIST}" \
  --input_root "${IMG_ROOT}" \
  --output_root "${SEG_OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --shape "${HEIGHT}" "${WIDTH}"

# Quick sanity: ensure some seg results exist
if ! find "${SEG_OUTPUT_DIR}" -type f -name "*.png" -o -name "*.npy" | grep -q . ; then
  echo "[seg_then_depth.sh] ERROR: No segmentation outputs found under ${SEG_OUTPUT_DIR}."
  echo "  Check your seg checkpoint/path and vis_seg1.py run."
  exit 2
fi

# -----------------------------------------------
# STEP 2: DEPTH (views 0..4) using seg outputs
# -----------------------------------------------
echo "[seg_then_depth.sh] === DEPTH ==="
echo "  checkpoint: ${DEPTH_CHECKPOINT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${DEPTH_RUN_FILE}" \
  "${DEPTH_CHECKPOINT}" \
  --input "${INPUT_LIST}" \
  --input_root "${IMG_ROOT}" \
  --seg_dir "${SEG_OUTPUT_DIR}" \
  --output_root "${DEPTH_OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --shape "${HEIGHT}" "${WIDTH}"

# -----------------------------------------------
# CLEANUP
# -----------------------------------------------
rm -f "${INPUT_LIST}"
echo "[seg_then_depth.sh] ✅ Done."
echo "  Seg saved under:   ${SEG_OUTPUT_DIR}/<object>_<cam>/"
echo "  Depth saved under: ${DEPTH_OUTPUT_DIR}/<object>_<cam>/"
