#!/bin/bash
set -euo pipefail

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE='torchscript'   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"

# ---------------------------- INPUT / OUTPUT ----------------------------------
SPLIT="val"
MODEL_NAME='sapiens_1b'
IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/${SPLIT}/img"
SEG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/seg/sapiens_0.3b"  #${MODEL_NAME}}"
OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/depth/${MODEL_NAME}"
RES_FILTER=""   # optional resolution filter like "1024x1024"

# ---------------------------- MODEL -------------------------------------------

# CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/depth/checkpoints/${MODEL_NAME}/${MODEL_NAME}_render_people_epoch_100_${MODE}.pt2"
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/depth/checkpoints/${MODEL_NAME}/sapiens_1b_render_people_epoch_88_$MODE.pt2"


# ---------------------------- RUNTIME / GPUS ----------------------------------
RUN_FILE='/home/clark/Documents/GitHub/sapiens/lite/demo/vis_depth1.py'
VALID_GPU_IDS=(0)
TOTAL_GPUS=1
BATCH_SIZE=8

# ---------------------------- PREPARE INPUT LIST -----------------------------
IMAGE_LIST="${IMG_ROOT}/image_list_all_0jpg.txt"

find "${IMG_ROOT}" -type f \( -name "0.jpg" -o -name "0.png" \) | sort > "${IMAGE_LIST}.candidates"

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

if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found after filtering. Check IMG_ROOT and RES_FILTER."
  exit 1
fi

# ---------------------------- RUN ---------------------------------------------
CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} python "${RUN_FILE}" \
  "${CHECKPOINT}" \
  --input "${IMG_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --seg_dir "${SEG_ROOT}" \
  --batch_size "${BATCH_SIZE}" \

rm -f "${IMAGE_LIST}"

cd - >/dev/null
echo "[DONE] Depth saved to ${OUTPUT_ROOT}"
