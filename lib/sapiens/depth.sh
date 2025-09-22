#!/bin/bash
set -euo pipefail

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE='torchscript'   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"

# ---------------------------- INPUTS / OUTPUTS --------------------------------
# THuman image root with many "<object>_<camera>" subfolders, each containing "0.jpg"
IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/mini-train/img"

# Segmentation root that ALSO mirrors the same hierarchy (each has 0004_000/, 0004_001/, ...)
# e.g. produced by the hierarchy-preserving seg script we just made:
SEG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/seg/sapiens_0.3b"

# Depth outputs root (we append model name and rel_dir under this)
OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/depth"

# Optional resolution filter: "WxH" (e.g., "1024x1024"); leave empty "" to disable
RES_FILTER=""

# ---------------------------- MODEL -------------------------------------------
MODEL_NAME='sapiens_0.3b'
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/depth/checkpoints/${MODEL_NAME}/${MODEL_NAME}_render_people_epoch_100_${MODE}.pt2"

OUTPUT="${OUTPUT_ROOT}/${MODEL_NAME}"
mkdir -p "${OUTPUT}"

# ---------------------------- RUNTIME / GPUS ----------------------------------
RUN_FILE='/home/clark/Documents/GitHub/sapiens/lite/demo/vis_depth.py'
TOTAL_GPUS=1
VALID_GPU_IDS=(0)

echo "[vis_depth] IMG_ROOT=${IMG_ROOT}"
echo "[vis_depth] SEG_ROOT=${SEG_ROOT}"
echo "[vis_depth] OUTPUT=${OUTPUT}"
echo "[vis_depth] RES_FILTER=${RES_FILTER:-<none>}"
echo "[vis_depth] CHECKPOINT=${CHECKPOINT}"
echo "[vis_depth] RUN_FILE=${RUN_FILE}"

# ---------------------------- Build IMAGE_LIST --------------------------------
# Collect ONLY 0.jpg (or 0.png) across all subfolders
IMAGE_LIST="${IMG_ROOT}/image_list_all_0jpg.txt"

# 1) gather candidates
find "${IMG_ROOT}" -type f \( -name "0.jpg" -o -name "0.png" \) | sort > "${IMAGE_LIST}.candidates"

# 2) optional resolution filter (Pillow-based; no ImageMagick needed)
if [ -n "${RES_FILTER:-}" ]; then
  echo "Filtering images by resolution: ${RES_FILTER}"
  export RES_FILTER IMAGE_LIST
  python - <<'PY'
import os
from PIL import Image

want = os.environ["RES_FILTER"].strip()           # e.g., "1024x1024"
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

# ---------------------------- Run (preserve hierarchy + paired seg) -----------
TMP_LIST_DIR="${OUTPUT}/.tmp_lists"
mkdir -p "${TMP_LIST_DIR}"

GPU_IDX=0
while IFS= read -r img; do
  # Relative folder like "0004_000" from ".../train/img/0004_000/0.jpg"
  rel_dir="$(dirname "${img#"${IMG_ROOT}/"}")"

  # Output dir mirrors the dataset hierarchy
  out_dir="${OUTPUT}/${rel_dir}"
  mkdir -p "${out_dir}"

  # Matching segmentation folder (must exist if you want seg-guided depth)
  seg_dir="${SEG_ROOT}/${rel_dir}"
  if [ ! -d "${seg_dir}" ]; then
    echo "[warn] Missing segmentation dir for ${rel_dir}: ${seg_dir}"
  fi

  # `vis_depth.py` expects a directory or a text file of images -> make a one-line list
  list_file="${TMP_LIST_DIR}/${rel_dir}.txt"
  mkdir -p "$(dirname "${list_file}")"
  printf '%s\n' "${img}" > "${list_file}"

  echo "[vis_depth] ${rel_dir} -> out=${out_dir}, seg=${seg_dir}"
  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[$GPU_IDX]} python "${RUN_FILE}" \
    "${CHECKPOINT}" \
    --input "${list_file}" \
    --seg_dir "${seg_dir}" \
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
