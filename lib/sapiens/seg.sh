#!/bin/bash
set -euo pipefail

cd ../../.. || exit

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE='torchscript'   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"

IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/train/img"
OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/seg"
MODEL_NAME='sapiens_0.3b'
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/checkpoints/${MODEL_NAME}/${MODEL_NAME}_goliath_best_goliath_mIoU_7673_epoch_194_${MODE}.pt2"

RUN_FILE="/home/clark/Documents/GitHub/sapiens/lite/demo/vis_seg1.py"
INPUT_LIST="${IMG_ROOT}/image_list_all_0jpg.txt"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"

BATCH_SIZE=16
GPU_ID=0

# -----------------------------------------------
# BUILD FLAT IMAGE LIST
# -----------------------------------------------
echo "[seg.sh] Scanning for all '0.jpg' or '0.png' in ${IMG_ROOT}"
find "${IMG_ROOT}" -type f \( -name "0.jpg" -o -name "0.png" \) | sort > "${INPUT_LIST}"

if [ ! -s "${INPUT_LIST}" ]; then
  echo "No images found. Check IMG_ROOT=${IMG_ROOT}"
  exit 1
fi

echo "[seg.sh] Found $(wc -l < "${INPUT_LIST}") images"
echo "[seg.sh] Output will be stored in ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------
# RUN
# -----------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${RUN_FILE}" \
  "${CHECKPOINT}" \
  --input "${INPUT_LIST}" \
  --input_root "${IMG_ROOT}" \
  --output_root "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --shape 1024 768

# -----------------------------------------------
# CLEANUP
# -----------------------------------------------
rm -f "${INPUT_LIST}"
echo "[seg.sh] ✅ Done. Output saved under: ${OUTPUT_DIR}/<object>_<cam>/"
