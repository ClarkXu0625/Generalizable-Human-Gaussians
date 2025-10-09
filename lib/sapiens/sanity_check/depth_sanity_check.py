#!/usr/bin/env python3
import os, json, argparse
import numpy as np

# Try to use matplotlib for figures; if missing, we skip plots gracefully.
_have_mpl = True
try:
    import matplotlib.pyplot as plt
except Exception:
    _have_mpl = False

def _ensuredir(p):
    os.makedirs(p, exist_ok=True); return p

def load_KRt(intrinsic_path, extrinsic_path):
    K = np.load(intrinsic_path).astype(np.float64)
    ext = np.load(extrinsic_path).astype(np.float64)
    R = ext[:3,:3].copy()
    t = ext[:3, 3:4].copy()  # (3,1)
    return K, R, t

def basic_stats(name, D):
    finite = np.isfinite(D)
    nz = np.count_nonzero(finite)
    N = D.size
    stats = {
        "shape": list(D.shape),
        "dtype": str(D.dtype),
        "finite_fraction": float(nz / max(N,1)),
        "num_zero": int(np.count_nonzero((D==0) & finite)),
        "num_negative": int(np.count_nonzero((D<0) & finite)),
        "num_nan": int(np.count_nonzero(~np.isfinite(D))),
        "min": float(np.nanmin(D)) if nz else None,
        "p01": float(np.nanpercentile(D, 1)) if nz else None,
        "p10": float(np.nanpercentile(D,10)) if nz else None,
        "median": float(np.nanmedian(D)) if nz else None,
        "p90": float(np.nanpercentile(D,90)) if nz else None,
        "p99": float(np.nanpercentile(D,99)) if nz else None,
        "max": float(np.nanmax(D)) if nz else None,
    }
    return stats

def save_heatmap(path, D, title, vmin=None, vmax=None):
    if not _have_mpl: return
    plt.figure(figsize=(6,5))
    im = plt.imshow(D, vmin=vmin, vmax=vmax)
    plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_hist(path, D, title, bins=100):
    if not _have_mpl: return
    vals = D[np.isfinite(D)]
    if vals.size == 0:
        return
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.title(title); plt.xlabel("depth"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def spearman_corr(x, y):
    # rank-based correlation (simple, no ties correction), subsample for speed
    m = min(x.size, 200_000)
    if x.size > m:
        idx = np.random.default_rng(0).choice(x.size, m, replace=False)
        x, y = x[idx], y[idx]
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12)
    return float((rx @ ry) / denom)

def robust_affine(x_rel, y_abs, tau=0.01, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x_rel) & np.isfinite(y_abs)
    x = x_rel[mask]; y = y_abs[mask]
    if x.size < 20:
        return {"alpha": None, "beta": None, "inliers": 0, "pairs": int(x_rel.size)}
    best_inl = np.array([], dtype=int); best_ab = (1.0, 0.0)
    for _ in range(iters):
        if x.size < 2: break
        i, j = rng.choice(x.size, 2, replace=False)
        if abs(x[j]-x[i]) < 1e-12: continue
        a = (y[j]-y[i])/(x[j]-x[i]); b = y[i] - a*x[i]
        pred = a*x + b
        inl = np.where(np.abs(pred - y) < tau)[0]
        if inl.size > best_inl.size:
            best_inl = inl; best_ab = (a, b)
    if best_inl.size >= 10:
        X = np.vstack([x[best_inl], np.ones(best_inl.size)]).T
        ab, *_ = np.linalg.lstsq(X, y[best_inl], rcond=None)
        a, b = float(ab[0]), float(ab[1])
        resid = (a*x[best_inl]+b) - y[best_inl]
        mae = float(np.mean(np.abs(resid)))
        return {"alpha": a, "beta": b, "inliers": int(best_inl.size), "pairs": int(x.size), "mae_inliers_m": mae}
    # fallback plain LS
    X = np.vstack([x, np.ones(x.size)]).T
    ab, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(ab[0]), float(ab[1])
    pred = a*x + b
    mae = float(np.mean(np.abs(pred - y)))
    return {"alpha": a, "beta": b, "inliers": int(x.size), "pairs": int(x.size), "mae_inliers_m": mae}

def backproject(K, R, t, u, v, d):
    uv1 = np.stack([u, v, np.ones_like(u)], axis=0).astype(np.float64)
    Kinv = np.linalg.inv(K)
    x_cam = Kinv @ (uv1 * d)
    Xw = (R.T @ (x_cam - t)).T
    return Xw

def project(K, R, t, Xw):
    Xc = (R @ Xw.T) + t
    u = (K[0,0]*Xc[0] + K[0,2]*Xc[2]) / (Xc[2] + 1e-12)
    v = (K[1,1]*Xc[1] + K[1,2]*Xc[2]) / (Xc[2] + 1e-12)
    return u, v

def extrinsic_convention_sanity(K, R, t, H, W):
    """
    We try both:
      A) world->cam: Xc = R Xw + t; backproj: Xw = R^T (Xc - t)
      B) cam->world: Xw = R Xc + t; equivalent world->cam: Xc = R^T (Xw - t)
    We test with a synthetic grid at a plausible Z and see which roundtrips better.
    """
    # Make a small grid of pixels and use a plausible constant depth (2m)
    ys = np.linspace(0, H-1, 9).astype(int)
    xs = np.linspace(0, W-1, 12).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    u = xx.ravel().astype(np.float64); v = yy.ravel().astype(np.float64)
    d = np.full_like(u, 2.0, dtype=np.float64)

    # A) assume ext is world->cam (our default)
    Kinv = np.linalg.inv(K)
    Xc = (Kinv @ (np.stack([u, v, np.ones_like(u)],0)*d)).astype(np.float64)
    Xw_A = (R.T @ (Xc - t)).T
    uA, vA = project(K, R, t, Xw_A)
    errA = float(np.mean(np.sqrt((uA-u)**2 + (vA-v)**2)))

    # B) assume ext is cam->world
    Xw_B = (R @ Xc + t).T
    # equivalent world->cam for projection:
    uB, vB = project(K, R.T, -R.T@t, Xw_B)
    errB = float(np.mean(np.sqrt((uB-u)**2 + (vB-v)**2)))

    return {"pixel_reproj_rmse_A": errA, "pixel_reproj_rmse_B": errB,
            "likely_convention": "world_to_cam(R,t)" if errA <= errB else "cam_to_world(R,t)"}

def main():
    ap = argparse.ArgumentParser(description="Sanity checks for SMPL absolute depth vs Sapiens relative depth (per view).")
    ap.add_argument("--id", required=True, help="e.g. 0004")
    ap.add_argument("--views", nargs=3, required=True, help="e.g. 000 006 011")
    ap.add_argument("--smpl_root",    required=True, help="datasets/THuman/train/depth")
    ap.add_argument("--sapiens_root", required=True, help="output/train/depth/sapiens_0.3b")
    ap.add_argument("--parm_root",    required=True, help="datasets/THuman/train/parm")
    ap.add_argument("--stride", type=int, default=4, help="subsample stride for scatter/affine")
    ap.add_argument("--tau", type=float, default=0.01, help="affine inlier tau (m)")
    ap.add_argument("--out_dir", default="sanity_out")
    args = ap.parse_args()

    _ensuredir(args.out_dir)
    perview_dir = _ensuredir(os.path.join(args.out_dir, "per_view"))

    report = {"id": args.id, "views": [], "notes": []}

    for view in args.views:
        folder = f"{args.id}_{view}"
        smpl_path = os.path.join(args.smpl_root,    folder, "0_smpl_depth_dbg0.npy")
        sapi_path = os.path.join(args.sapiens_root, folder, "0.npy")
        intr_path = os.path.join(args.parm_root,    folder, "0_intrinsic.npy")
        extr_path = os.path.join(args.parm_root,    folder, "0_extrinsic.npy")

        for p in (smpl_path, sapi_path, intr_path, extr_path):
            if not os.path.exists(p):
                report["notes"].append(f"[{folder}] missing: {p}")
        if not all(os.path.exists(p) for p in (smpl_path, sapi_path, intr_path, extr_path)):
            continue

        d_abs = np.load(smpl_path).astype(np.float64)
        d_rel = np.load(sapi_path).astype(np.float64)
        if d_abs.ndim == 3: d_abs = d_abs.squeeze()
        if d_rel.ndim == 3: d_rel = d_rel.squeeze()

        K, R, t = load_KRt(intr_path, extr_path)
        H, W = d_abs.shape

        # Basic stats + images/hists
        smpl_stats = basic_stats("smpl_abs", d_abs)
        sapi_stats = basic_stats("sapi_rel", d_rel)

        # depth heatmaps (use same vmin/vmax window from SMPL percentiles for fair viz)
        if smpl_stats["p01"] is not None and smpl_stats["p99"] is not None:
            vmin = smpl_stats["p01"]; vmax = smpl_stats["p99"]
        else:
            vmin = None; vmax = None
        save_heatmap(os.path.join(perview_dir, f"{folder}_smpl_abs.png"), d_abs, f"{folder} SMPL abs (m)", vmin, vmax)
        save_heatmap(os.path.join(perview_dir, f"{folder}_sapi_rel_same_scale.png"), d_rel, f"{folder} Sapiens rel (same scale)", vmin, vmax)
        # separate autoscaled view for Sapiens (to see its own distribution)
        save_heatmap(os.path.join(perview_dir, f"{folder}_sapi_rel_auto.png"), d_rel, f"{folder} Sapiens rel (auto)")

        save_hist(os.path.join(perview_dir, f"{folder}_smpl_abs_hist.png"), d_abs, f"{folder} SMPL abs (hist)")
        save_hist(os.path.join(perview_dir, f"{folder}_sapi_rel_hist.png"), d_rel, f"{folder} Sapiens rel (hist)")

        # Overlap mask and scatter
        valid = np.isfinite(d_abs) & (d_abs > 0) & np.isfinite(d_rel) & (d_rel > 0)
        frac_overlap = float(np.count_nonzero(valid) / d_abs.size)
        ys, xs = np.where(valid)
        if args.stride > 1 and ys.size > 0:
            sel = np.arange(0, ys.size, args.stride)
            ys, xs = ys[sel], xs[sel]
        dA = d_abs[ys, xs]; dR = d_rel[ys, xs]

        # Spearman (monotonic) + robust affine fit
        sp = spearman_corr(dR, dA) if dA.size > 10 else None
        aff = robust_affine(dR, dA, tau=args.tau, iters=2000, seed=0)

        # Optional scatter plot
        if _have_mpl and dA.size > 10:
            plt.figure(figsize=(5,5))
            plt.scatter(dR, dA, s=1, alpha=0.3)
            if aff["alpha"] is not None:
                xr = np.linspace(np.nanmin(dR), np.nanmax(dR), 50)
                yr = aff["alpha"]*xr + aff["beta"]
                plt.plot(xr, yr, lw=2)
            plt.xlabel("Sapiens relative depth"); plt.ylabel("SMPL absolute depth (m)")
            plt.title(f"{folder} scatter (Spearman={sp:.3f} if not None)")
            plt.tight_layout()
            plt.savefig(os.path.join(perview_dir, f"{folder}_scatter_rel_vs_abs.png"), dpi=150)
            plt.close()

        # Extrinsics convention sanity
        conv = extrinsic_convention_sanity(K, R, t, H, W)

        view_rep = {
            "folder": folder,
            "smpl_abs_stats": smpl_stats,
            "sapi_rel_stats": sapi_stats,
            "overlap_fraction": frac_overlap,
            "spearman_rel_vs_abs": sp,
            "affine_fit_rel_to_abs": aff,
            "extrinsic_convention_check": conv
        }
        report["views"].append(view_rep)

    # Write JSON report
    with open(os.path.join(args.out_dir, "sanity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Wrote sanity report to {os.path.join(args.out_dir,'sanity_report.json')}")
    print(f"Per-view figures in: {perview_dir}")
    if not _have_mpl:
        print("[NOTE] matplotlib not installed; skipped plots (JSON stats still computed).")

if __name__ == "__main__":
    main()
