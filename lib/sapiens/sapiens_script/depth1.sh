#!/bin/bash
set -euo pipefail

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT="/home/${USER}/Documents/GitHub/sapiens/lite"
MODE='torchscript'   # or 'bfloat16'
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT}/${MODE}"

# ---------------------------- INPUT / OUTPUT ----------------------------------
SPLIT="train"
MODEL_NAME='sapiens_1b'
IMG_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/${SPLIT}/img"
SEG_ROOT_BASE="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/seg/sapiens_1b"
OUTPUT_ROOT="/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/${SPLIT}/depth/${MODEL_NAME}"

# ---------------------------- MODEL -------------------------------------------
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/depth/checkpoints/${MODEL_NAME}/sapiens_1b_render_people_epoch_88_${MODE}.pt2"

# ---------------------------- RUNTIME -----------------------------------------
RUN_FILE='/home/clark/Documents/GitHub/sapiens/lite/demo/vis_depth1.py'
GPU_ID=0
BATCH_SIZE=8                  # ← as requested
USE_FP16=${USE_FP16:-0}       # optional: set to 1 to pass --fp16

# ---------------------------- OPTIONAL LIMIT ----------------------------------
# ARG_ID can be:
#   ""           → all views
#   "0004"       → all 16 views for object 0004
#   "0004_007"   → only that single view
ARG_ID="${1:-}"

if [ -z "${ARG_ID}" ]; then
  mapfile -t VIEW_DIRS < <(find "${IMG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_*" | sort)
elif [[ "${ARG_ID}" == *_* ]]; then
  VIEW="${IMG_ROOT}/${ARG_ID}"
  [ -d "${VIEW}" ] || { echo "ERROR: View not found: ${VIEW}"; exit 2; }
  VIEW_DIRS=( "${VIEW}" )
else
  mapfile -t VIEW_DIRS < <(find "${IMG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${ARG_ID}_*" | sort)
  [ ${#VIEW_DIRS[@]} -gt 0 ] || { echo "ERROR: No views for object ${ARG_ID}"; exit 3; }
fi

# ---------------------------- STAGE + LIST ------------------------------------
STAGE_PARENT="/mnt/esdata/tmp"   # or another big mount
mkdir -p "${STAGE_PARENT}"
STAGE_BASE="$(mktemp -d -p "${STAGE_PARENT}" sapiens_depth_stage_XXXXXX)"
#STAGE_BASE="$(mktemp -d -p /tmp sapiens_depth_stage_XXXXXX)"
trap 'rm -rf "${STAGE_BASE}"' EXIT
STAGE_IMG="${STAGE_BASE}/img"   # vis_depth1 --input points here
STAGE_SEG="${STAGE_BASE}/seg"   # vis_depth1 --seg_dir points here
STAGE_OUT="${STAGE_BASE}/out"   # vis_depth1 --output_root writes here
mkdir -p "${STAGE_IMG}" "${STAGE_SEG}" "${STAGE_OUT}"

# temp TXT list of (view, subview) items (as requested)
LIST_TXT="${STAGE_BASE}/staged_list.txt"
: > "${LIST_TXT}"

echo "[depth_batch1] checkpoint: ${CHECKPOINT}"
echo "[depth_batch1] staging img: ${STAGE_IMG}"
echo "[depth_batch1] staging seg: ${STAGE_SEG}"
echo "[depth_batch1] output temp: ${STAGE_OUT}"
echo "[depth_batch1] views      : ${#VIEW_DIRS[@]}"

TOTAL=0
for VIEW_PATH in "${VIEW_DIRS[@]}"; do
  VIEW_NAME="$(basename "${VIEW_PATH}")"              # e.g., 0004_000
  SEG_VIEW_DIR="${SEG_ROOT_BASE}/${VIEW_NAME}"

  for S in 0 1 2 3 4; do
    # locate subview image
    SRC_IMG=""
    for EXT in jpg png jpeg; do
      CAND="${VIEW_PATH}/${S}.${EXT}"
      if [ -f "${CAND}" ]; then SRC_IMG="${CAND}"; SRC_EXT="${EXT}"; break; fi
    done
    if [ -z "${SRC_IMG}" ]; then
      echo "  [skip] ${VIEW_NAME} s${S}: missing ${S}.{jpg|png|jpeg}"
      continue
    fi

    # locate seg npy (optional)
    SRC_SEGN=""
    CSEGN="${SEG_VIEW_DIR}/${S}.npy"
    [ -f "${CSEGN}" ] && SRC_SEGN="${CSEGN}"

    PV_NAME="${VIEW_NAME}__s${S}"
    PV_IMG_DIR="${STAGE_IMG}/${PV_NAME}"
    PV_SEG_DIR="${STAGE_SEG}/${PV_NAME}"
    mkdir -p "${PV_IMG_DIR}" "${PV_SEG_DIR}"

    # symlink image as 0.<ext> and (if present) seg as 0.npy
    ln -snf "${SRC_IMG}" "${PV_IMG_DIR}/0.${SRC_EXT}"
    if [ -n "${SRC_SEGN}" ]; then
      ln -snf "${SRC_SEGN}" "${PV_SEG_DIR}/0.npy"
    fi

    # append to txt list (absolute paths)
    echo "${PV_IMG_DIR}/0.${SRC_EXT}  ${PV_SEG_DIR}/0.npy  ${VIEW_NAME}  ${S}" >> "${LIST_TXT}"

    TOTAL=$((TOTAL+1))
  done
done

[ ${TOTAL} -gt 0 ] || { echo "Nothing staged. Exiting."; exit 0; }
echo "[depth_batch1] staged pseudo-views: ${TOTAL}"
echo "[depth_batch1] list file: ${LIST_TXT}"

# Note: vis_depth1.py expects a directory (not a file list) as --input.
# We still create LIST_TXT for auditing, but pass the staging directory.
FP16_FLAG=""; [ "${USE_FP16}" = "1" ] && FP16_FLAG="--fp16"

# ---------------------------- SINGLE PROCESS CALL -----------------------------
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${RUN_FILE}" \
  "${CHECKPOINT}" \
  --input "${STAGE_IMG}" \
  --output_root "${STAGE_OUT}" \
  --seg_dir "${STAGE_SEG}" \
  --batch_size "${BATCH_SIZE}" \
  ${FP16_FLAG}

# ---------------------------- COLLECT BACK ------------------------------------
# vis_depth1.py writes per-pseudo-view outputs as:
#   ${STAGE_OUT}/${VIEW_NAME}__s${S}/0.{jpg|png|npy}
# Move them into:
#   ${OUTPUT_ROOT}/${VIEW_NAME}/${S}.{jpg|png|npy}
moved=0
for PV_DIR in "${STAGE_OUT}"/*; do
  [ -d "${PV_DIR}" ] || continue
  PV_NAME="$(basename "${PV_DIR}")"        # e.g., 0004_000__s3
  VIEW_NAME="${PV_NAME%%__s*}"             # 0004_000
  S="${PV_NAME##*__s}"                     # 3

  FINAL_DIR="${OUTPUT_ROOT}/${VIEW_NAME}"
  mkdir -p "${FINAL_DIR}"

  [ -f "${PV_DIR}/0.jpg" ] && { mv -f "${PV_DIR}/0.jpg" "${FINAL_DIR}/${S}.jpg"; moved=1; }
  [ -f "${PV_DIR}/0.png" ] && { mv -f "${PV_DIR}/0.png" "${FINAL_DIR}/${S}.png"; moved=1; }
  [ -f "${PV_DIR}/0.npy" ] && { mv -f "${PV_DIR}/0.npy" "${FINAL_DIR}/${S}.npy"; moved=1; }
done

echo "[DONE] Depth saved under ${OUTPUT_ROOT} (one-process; batch_size=${BATCH_SIZE})"
echo "       Staged list for reference: ${LIST_TXT}"
if [ "${moved}" -eq 0 ]; then
  echo "       (No files moved. If outputs landed directly under ${STAGE_OUT}, tell me and I'll tweak the collector.)"
fi
