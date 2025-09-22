#!/usr/bin/env python3
import os, sys, argparse, glob
import numpy as np
import cv2
from collections import defaultdict

def find_rel_dirs(root, pattern):
    """Recursively find *directories* (relative to root) that contain a file named 'pattern'."""
    hits = []
    for p in glob.glob(os.path.join(root, "**", pattern), recursive=True):
        d = os.path.dirname(p)
        rel = os.path.relpath(d, root)
        if rel == ".":
            rel = ""
        hits.append(rel)
    return sorted(set(hits))

def fit_scale_shift(x, y, m, robust=False, delta=0.05, iters=5):
    """
    x: relative depth (H,W), y: absolute depth (H,W), m: boolean mask (H,W)
    robust: Huber reweighting
    Returns s, t, stats dict
    """
    xv = x[m].astype(np.float64)
    yv = y[m].astype(np.float64)
    if xv.size < 10:
        return np.nan, np.nan, {"n": int(xv.size), "mae": np.nan, "rmse": np.nan, "med": np.nan, "s": np.nan, "t": np.nan}

    w = np.ones_like(xv)

    def solve_wls(xv, yv, w):
        Sw  = w.sum()
        Sx  = (w*xv).sum()
        Sy  = (w*yv).sum()
        Sxx = (w*xv*xv).sum()
        Sxy = (w*xv*yv).sum()
        denom = (Sw*Sxx - Sx*Sx)
        if abs(denom) < 1e-12:
            return np.nan, np.nan
        s = (Sw*Sxy - Sx*Sy)/denom
        t = (Sy - s*Sx)/Sw
        return s, t

    s, t = solve_wls(xv, yv, w)
    if robust:
        for _ in range(iters):
            r = (s*xv + t) - yv
            a = np.abs(r)
            w = np.ones_like(a)
            mask_h = a > delta
            w[mask_h] = delta/(a[mask_h] + 1e-12)
            s, t = solve_wls(xv, yv, w)
            if not np.isfinite(s) or not np.isfinite(t):
                break

    r = (s*xv + t) - yv
    stats = {
        "n": int(xv.size),
        "mae": float(np.mean(np.abs(r))) if xv.size else np.nan,
        "rmse": float(np.sqrt(np.mean(r**2))) if xv.size else np.nan,
        "med": float(np.median(r)) if xv.size else np.nan,
        "s": float(s), "t": float(t),
    }
    return float(s), float(t), stats

def colormap_depth(d, mask=None, qlo=2, qhi=98, invert=False):
    """Depth → colored uint8 BGR for quick sanity PNGs."""
    H, W = d.shape
    out = np.zeros((H, W, 3), np.uint8)
    if mask is None:
        mask = np.isfinite(d)
    valid = np.isfinite(d) & mask
    if not np.any(valid):
        return out
    v = d[valid].astype(np.float64)
    lo = np.percentile(v, qlo)
    hi = np.percentile(v, qhi)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    n = (v - lo)/(hi - lo)
    if invert:
        n = 1.0 - n
    n = np.clip(n, 0, 1)
    n8 = (n * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(n8, cv2.COLORMAP_INFERNO)
    out[valid] = cm.reshape((-1, 3))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl_root", required=True)
    ap.add_argument("--sapiens_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pattern_smpl", default="0_smpl_depth.npy",
                    help="SMPL depth filename inside each <obj>_<cam> folder")
    ap.add_argument("--pattern_sapiens", default="0.npy",
                    help="Sapiens depth filename inside each <obj>_<cam> folder")
    ap.add_argument("--seg_root", default=None,
                    help="Optional segmentation root; if provided, mask ∧= (seg>0).")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="SMPL background threshold (meters).")
    args = ap.parse_args()

    # find directories (relative) that contain the target file pattern
    smpl_dirs    = find_rel_dirs(args.smpl_root,    args.pattern_smpl)
    sapiens_dirs = find_rel_dirs(args.sapiens_root, args.pattern_sapiens)

    both = sorted(set(smpl_dirs) & set(sapiens_dirs))

    # helpful logs if empty or partial mismatch
    print(f"[scan] SMPL dirs with '{args.pattern_smpl}':     {len(smpl_dirs)}")
    print(f"[scan] SAPIENS dirs with '{args.pattern_sapiens}': {len(sapiens_dirs)}")
    print(f"[scan] Intersection: {len(both)}")

    if len(both) == 0:
        # show a few examples to help debug path issues
        only_smpl = sorted(set(smpl_dirs) - set(sapiens_dirs))[:10]
        only_sap  = sorted(set(sapiens_dirs) - set(smpl_dirs))[:10]
        if only_smpl:
            print("[hint] Examples only in SMPL:", only_smpl[:5])
        if only_sap:
            print("[hint] Examples only in SAPIENS:", only_sap[:5])
        print("No matching <obj>_<cam> folders found. Check roots/patterns/levels.")
        sys.exit(1)

    os.makedirs(args.out_root, exist_ok=True)
    log_lines = ["rel_dir,n,mae,rmse,med,s,t"]

    for rel in both:
        smpl_fp = os.path.join(args.smpl_root, rel, args.pattern_smpl)
        sap_fp  = os.path.join(args.sapiens_root, rel, args.pattern_sapiens)
        out_dir = os.path.join(args.out_root, rel)
        os.makedirs(out_dir, exist_ok=True)

        # load depths
        try:
            smpl = np.load(smpl_fp).astype(np.float32)  # meters, bg ~ 0
            sap  = np.load(sap_fp).astype(np.float32)   # relative
        except Exception as e:
            print(f"[{rel}] load error: {e}")
            continue

        # size align if needed
        if smpl.shape != sap.shape:
            sap = cv2.resize(sap, (smpl.shape[1], smpl.shape[0]), interpolation=cv2.INTER_LINEAR)

        # unified mask
        m = np.isfinite(sap)
        m &= (smpl > args.eps)
        if args.seg_root:
            seg_fp = os.path.join(args.seg_root, rel, "0.npy")
            if os.path.isfile(seg_fp):
                try:
                    seg = np.load(seg_fp)
                    if seg.shape != smpl.shape:
                        seg = cv2.resize(seg.astype(np.float32), (smpl.shape[1], smpl.shape[0]), interpolation=cv2.INTER_NEAREST)
                    m &= (seg > 0)
                except Exception as e:
                    print(f"[{rel}] seg load error: {e}")

        if not np.any(m):
            print(f"[{rel}] no valid overlap; skipping.")
            continue

        # fit s,t
        s, t, stats = fit_scale_shift(sap, smpl, m, robust=args.robust, delta=args.delta)
        print(f"[{rel}] n={stats['n']}  s={stats['s']:.6f}  t={stats['t']:.6f}  "
              f"MAE={stats['mae']:.5f}  RMSE={stats['rmse']:.5f}  MED={stats['med']:.5f}")
        log_lines.append(f"{rel},{stats['n']},{stats['mae']:.6f},{stats['rmse']:.6f},{stats['med']:.6f},{stats['s']:.8f},{stats['t']:.8f}")

        # align
        sap_aligned = s * sap + t
        sap_aligned[~m] = np.nan
        np.save(os.path.join(out_dir, "0_aligned.npy"), sap_aligned.astype(np.float32))

        # sanity images
        abs_cm  = colormap_depth(smpl, mask=m)
        aln_cm  = colormap_depth(sap_aligned, mask=m)
        res_cm  = colormap_depth(np.abs(sap_aligned - smpl), mask=m)
        vis = np.concatenate([abs_cm, aln_cm, res_cm], axis=1)
        cv2.imwrite(os.path.join(out_dir, "align_sanity.png"), vis)

    with open(os.path.join(args.out_root, "align_stats.csv"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

if __name__ == "__main__":
    main()
