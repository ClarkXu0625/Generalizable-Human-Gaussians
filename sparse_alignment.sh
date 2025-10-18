#!/bin/bash
set -euo pipefail

# ============================================================
# Usage examples:
#   ./depth_align_subviews.sh --obj all --view all --subview all
#   ./depth_align_subviews.sh --obj 0004 --view all --subview all
#   ./depth_align_subviews.sh --obj 0004 --view 0004_005 --subview 2
#   ./depth_align_subviews.sh --obj 0004 --view 0004_003 --subview all
# ============================================================

# -------------------- Default arguments ---------------------
OBJ="all"
VIEW="all"
SUBVIEW="all"

# -------------------- Parse CLI args ------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --obj)     OBJ="$2"; shift 2 ;;
    --view)    VIEW="$2"; shift 2 ;;
    --subview) SUBVIEW="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# -------------------- Paths & Constants ---------------------
PYTHON=python
SCRIPT="lib/sapiens/sparse_alignment.py"

SMPL_ROOT="datasets/THuman/train/depth"
SAPIENS_ROOT="lib/sapiens/output/train/depth/sapiens_1b"
OUT_ROOT="lib/sapiens/output/train/depth_aligned_clothed/sapiens_1b"
SEG_IMG_ROOT="datasets/THuman/train/mask"

ROBUST="--robust"
DELTA="--delta 0.05"

# -------------------- Determine which subviews ---------------
if [[ "${SUBVIEW}" == "all" ]]; then
  SUBVIEWS=(0 1 2 3 4)
else
  SUBVIEWS=("${SUBVIEW}")
fi

# -------------------- Resolve OBJ / VIEW → CAMS --------------
OBJ_FILTER=()
CAMS_FILTER=()

# If a specific VIEW like 0004_005 is given, force OBJ to its prefix and CAMS to that single cam.
if [[ "${VIEW}" != "all" && "${VIEW}" == *_* ]]; then
  VIEW_OBJ="${VIEW%_*}"
  VIEW_CAM="${VIEW##*_}"
  if [[ "${OBJ}" == "all" ]]; then
    OBJ="${VIEW_OBJ}"
  fi
  OBJ_FILTER=(--obj "${OBJ}")
  CAMS_FILTER=(--cams "${VIEW_CAM}")
else
  # VIEW == all. If OBJ is specified, enumerate its cams from the filesystem.
  if [[ "${OBJ}" != "all" ]]; then
    OBJ_FILTER=(--obj "${OBJ}")
    mapfile -t CAMS_LIST < <(find "${SAPIENS_ROOT}" -maxdepth 1 -type d -name "${OBJ}_*" -printf "%f\n" | awk -F'_' '{print $2}' | sort)
    if [[ ${#CAMS_LIST[@]} -eq 0 ]]; then
      echo "[align] ERROR: No views found for object ${OBJ} under ${SAPIENS_ROOT}"
      exit 2
    fi
    # >>> Minimal fix: pass ONE --cams followed by ALL cams (nargs='+')
    CAMS_FILTER=(--cams "${CAMS_LIST[@]}")
  fi
fi

# -------------------- Log the plan ---------------------------
echo "───────────────────────────────────────────────"
echo "[align] smpl_root     : ${SMPL_ROOT}"
echo "[align] sapiens_root  : ${SAPIENS_ROOT}"
echo "[align] out_root      : ${OUT_ROOT}"
echo "[align] seg_img_root  : ${SEG_IMG_ROOT}"
echo "[align] obj           : ${OBJ}"
echo "[align] view          : ${VIEW}"
if [[ ${#CAMS_FILTER[@]} -gt 0 ]]; then
  # pretty-print cams when present
  printf "[align] cams          : "; printf "%s " "${CAMS_FILTER[@]:1}"; printf "\n"
else
  echo "[align] cams          : (all)"
fi
echo "[align] subviews      : ${SUBVIEWS[*]}"
echo "───────────────────────────────────────────────"

# -------------------- Run alignment --------------------------
for S in "${SUBVIEWS[@]}"; do
  echo "[align] Processing subview ${S} ..."
  PATTERN_SMPL="${S}_smpl_depth.npy"
  PATTERN_SAPIENS="${S}.npy"
  SEG_IMG_NAME="${S}.png"

  ${PYTHON} "${SCRIPT}" \
    --smpl_root "${SMPL_ROOT}" \
    --sapiens_root "${SAPIENS_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --pattern_smpl "${PATTERN_SMPL}" \
    --pattern_sapiens "${PATTERN_SAPIENS}" \
    --seg_img_root "${SEG_IMG_ROOT}" \
    --seg_img_name "${SEG_IMG_NAME}" \
    --subview "${S}" \
    ${ROBUST} ${DELTA} \
    "${OBJ_FILTER[@]}" \
    "${CAMS_FILTER[@]}"
done

echo "[DONE] Depth alignment finished for requested subviews."
