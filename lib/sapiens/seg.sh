#!/bin/bash
set -euo pipefail

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE='torchscript'   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"

# ---------------------------- INPUT / OUTPUT ----------------------------------
IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/mini-train/img"
OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/seg"
RES_FILTER=""   # e.g. "1024x1024" to filter; empty to disable

# ---------------------------- MODEL -------------------------------------------
MODEL_NAME='sapiens_0.3b'
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/checkpoints/${MODEL_NAME}/${MODEL_NAME}_goliath_best_goliath_mIoU_7673_epoch_194_${MODE}.pt2"

OUTPUT="${OUTPUT_ROOT}/${MODEL_NAME}"
mkdir -p "${OUTPUT}"

# ---------------------------- RUNTIME / GPUS ----------------------------------
RUN_FILE='/home/clark/Documents/GitHub/sapiens/lite/demo/vis_seg.py'
TOTAL_GPUS=1
VALID_GPU_IDS=(0)

echo "[vis_seg] IMG_ROOT=${IMG_ROOT}"
echo "[vis_seg] OUTPUT=${OUTPUT}"
echo "[vis_seg] RES_FILTER=${RES_FILTER:-<none>}"
echo "[vis_seg] CHECKPOINT=${CHECKPOINT}"
echo "[vis_seg] RUN_FILE=${RUN_FILE}"

# ---------------------------- Build IMAGE_LIST --------------------------------
IMAGE_LIST="${IMG_ROOT}/image_list_all_0jpg.txt"

# 1) gather candidates (only 0.jpg / 0.png)
find "${IMG_ROOT}" -type f \( -name "0.jpg" -o -name "0.png" \) | sort > "${IMAGE_LIST}.candidates"

# 2) optional resolution filter (Pillow-based; no ImageMagick)
if [ -n "${RES_FILTER:-}" ]; then
  echo "Filtering images by resolution: ${RES_FILTER}"
  export RES_FILTER IMAGE_LIST
  python - <<'PY'
import os
from PIL import Image

want = os.environ["RES_FILTER"].strip()
cands = os.environ["IMAGE_LIST"] + ".candidates"
out   = os.environ["IMAGE_LIST"]

keep = []
with open(cands, "r") as f:
    for p in f:
        p = p.strip()
        if not p: continue
        try:
            with Image.open(p) as im:
                if f"{im.width}x{im.height}" == want:
                    keep.append(p)
        except Exception:
            pass

keep.sort()
with open(out, "w") as w:
    w.write("\n".join(keep))
print(f"[filter] kept {len(keep)} / {sum(1 for _ in open(cands))} -> {out}")
PY
else
  mv "${IMAGE_LIST}.candidates" "${IMAGE_LIST}"
fi
rm -f "${IMAGE_LIST}.candidates" || true

# Guard
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found after filtering. Check IMG_ROOT='${IMG_ROOT}' and RES_FILTER='${RES_FILTER}'."
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=2

# ---------------------------- Run (preserve hierarchy) ------------------------
TMP_LIST_DIR="${OUTPUT}/.tmp_lists"
mkdir -p "${TMP_LIST_DIR}"

GPU_IDX=0
while IFS= read -r img; do
  # Relative folder like "0004_000" from ".../train/img/0004_000/0.jpg"
  rel_dir="$(dirname "${img#"${IMG_ROOT}/"}")"
  out_dir="${OUTPUT}/${rel_dir}"
  mkdir -p "${out_dir}"

  # One-line temp list file for vis_seg.py
  list_file="${TMP_LIST_DIR}/${rel_dir}.txt"
  mkdir -p "$(dirname "${list_file}")"
  printf '%s\n' "${img}" > "${list_file}"

  echo "[vis_seg] ${rel_dir} -> ${out_dir}"
  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[$GPU_IDX]} python "${RUN_FILE}" \
    "${CHECKPOINT}" \
    --input "${list_file}" \
    --batch-size 1 \
    --output-root "${out_dir}"

  # round-robin GPUs if you later set TOTAL_GPUS>1
  GPU_IDX=$(( (GPU_IDX + 1) % TOTAL_GPUS ))
done < "${IMAGE_LIST}"

# ---------------------------- Cleanup -----------------------------------------
rm -f "${IMAGE_LIST}"
rm -rf "${TMP_LIST_DIR}"

cd - >/dev/null
echo "Processing complete."
echo "Results saved under ${OUTPUT}/<object>_<camera>/"
