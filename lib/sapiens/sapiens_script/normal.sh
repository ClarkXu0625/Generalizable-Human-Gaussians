#!/bin/bash
set -euo pipefail

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE="torchscript"   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"
SPLIT="train"

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.3b/sapiens_0.3b_normal_render_people_epoch_66_$MODE.pt2
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.6b/sapiens_0.6b_normal_render_people_epoch_200_$MODE.pt2
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115_$MODE.pt2
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_$MODE.pt2

# ----------------- INPUTS -----------------
# Root with many "<obj>_<cam>" folders; we will take only 0.jpg / 0.png in each.
IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/${SPLIT}/img"
# Nested segmentation root (same structure as IMG_ROOT, saves from vis_seg.py)
SEG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/seg/${MODEL_NAME}"
# Output (nested preserved)
OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/normal"


RUN_FILE="/home/clark/Documents/GitHub/sapiens/lite/demo/vis_normal1.py"
BATCH_SIZE=8
TOTAL_GPUS=1
VALID_GPU_IDS=(0)

echo "[vis_normal] IMG_ROOT=${IMG_ROOT}"
echo "[vis_normal] SEG_ROOT=${SEG_ROOT}"
echo "[vis_normal] OUTPUT_ROOT=${OUTPUT_ROOT}/${MODEL_NAME}"
echo "[vis_normal] CHECKPOINT=${CHECKPOINT}"
echo "[vis_normal] RUN_FILE=${RUN_FILE}"

OUT_MODEL_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"
mkdir -p "${OUT_MODEL_DIR}"

# Build a flat list of ONLY the desired images (0.jpg / 0.png)
IMAGE_LIST="${OUT_MODEL_DIR}/image_list_0_only.txt"
find "${IMG_ROOT}" -type f \( -name "0.jpg" -o -name "0.png" \) | sort > "${IMAGE_LIST}"

if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found (0.jpg/0.png). Check IMG_ROOT."
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=2
GPU_IDX=0
CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[$GPU_IDX]} \
python "${RUN_FILE}" \
  "${CHECKPOINT}" \
  --input "${IMAGE_LIST}" \
  --input_root "${IMG_ROOT}" \
  --seg_dir "${SEG_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --output_root "${OUT_MODEL_DIR}"

rm -f "${IMAGE_LIST}"

echo "Done. Results under ${OUT_MODEL_DIR}/<obj>_<cam>/{0.npy,0.png}"
