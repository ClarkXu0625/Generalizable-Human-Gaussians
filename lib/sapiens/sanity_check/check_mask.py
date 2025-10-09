#!/usr/bin/env python3
import os, argparse, glob
import numpy as np
import cv2

def load_mask_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        m = np.load(path)
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(np.float32), "npy"
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read: {path}")
        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA → use alpha channel as mask
            m = img[..., 3].astype(np.float32)
            src = "image:alpha"
        elif img.ndim == 3:
            m = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            src = "image:gray"
        else:
            m = img.astype(np.float32)
            src = "image:mono"
        return m, src

def summarize_mask(m, name, thr=1):
    H, W = m.shape[:2]
    vmin, vmax = float(np.min(m)), float(np.max(m))
    uniq = np.unique(m)
    uniq_sample = uniq[:10] if uniq.size > 10 else uniq
    num_mid = int(np.sum((m > 0) & (m < 255)))
    is_binary_0255 = np.array_equal(np.unique(m.astype(np.uint8)), np.array([0, 255])) or (num_mid == 0 and (vmin in (0,1)) and vmax in (1,255))
    fg_ratio = float(np.mean(m > thr))
    bg_ratio = 1.0 - fg_ratio

    print(f"\n[{name}] shape={H}x{W} dtype={m.dtype}")
    print(f"  min={vmin:.1f} max={vmax:.1f}  unique_count={uniq.size}")
    if uniq.size <= 10:
        print(f"  unique values: {uniq_sample}")
    else:
        print(f"  first 10 uniques: {uniq_sample} ...")
    print(f"  mid-gray (0<val<255) pixels: {num_mid}")
    print(f"  looks_binary_0_255: {is_binary_0255}")
    print(f"  fg(> {thr}) ratio: {fg_ratio:.3f}  bg ratio: {bg_ratio:.3f}")

    if is_binary_0255:
        if fg_ratio < 0.05 or fg_ratio > 0.95:
            print("  ⚠ mask is extremely unbalanced (check polarity or crop).")
        if fg_ratio > 0.5:
            print("  hint: if background is white in the image, you may need --invert_mask.")
        else:
            print("  hint: typical human fg≈0.15–0.35 depending on crop; this looks reasonable." if 0.05 < fg_ratio < 0.5 else "")
    else:
        print("  ⚠ not strictly binary → likely anti-aliased edges or soft mask.")
        print("    try: --mask_thr 180~220 (stricter) and/or morphology: --morph_close 3 --morph_open 2")

def main():
    ap = argparse.ArgumentParser(description="Quick sanity check for mask images/NPYs.")
    ap.add_argument("paths", nargs="+", help="Mask file(s) or globs (e.g., path/to/*/mask.png)")
    ap.add_argument("--thr", type=int, default=1, help="Threshold used for fg ratio (default: >1 is foreground).")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    if not files:
        print("No files matched."); return

    for fp in sorted(files):
        try:
            m, src = load_mask_any(fp)
            summarize_mask(m, f"{fp} ({src})", thr=args.thr)
        except Exception as e:
            print(f"[{fp}] ERROR: {e}")

if __name__ == "__main__":
    main()
