#!/usr/bin/env python3
import argparse, os, glob
import numpy as np
import cv2
import open3d as o3d

# ----------------------------
# Your original helpers (unchanged math)
# ----------------------------
def build_intrinsics(W, H, f=None, fov_deg=None):
    if f is None:
        if fov_deg is None:
            fov_deg = 60.0
        f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

def backproject_to_camera(Z_m, K):
    H, W = Z_m.shape
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)  # HxW

    Zf = Z_m.reshape(-1)
    uf = u.reshape(-1)
    vf = v.reshape(-1)

    good = np.isfinite(Zf) & (Zf > 0)
    Zf = Zf[good]; uf = uf[good]; vf = vf[good]

    X = (uf - cx) / fx * Zf
    Y = (vf - cy) / fy * Zf
    pts = np.stack([X, Y, Zf], axis=1)  # Nx3
    return pts, good.reshape(H, W)

def extract_colors(rgb_path, target_shape, valid_mask_flat):
    if (rgb_path is None) or (not os.path.exists(rgb_path)):
        return None
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    if img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = (rgb.reshape(-1, 3).astype(np.float32) / 255.0)[valid_mask_flat]
    return rgb

def load_mask_from_image(mask_img_path, target_shape):
    img = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read mask image: {mask_img_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]; m = alpha > 0
    else:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        m = gray > 0
    if m.shape[:2] != target_shape:
        m = cv2.resize(m.astype(np.uint8), (target_shape[1], target_shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m

def load_mask_from_npy(mask_npy_path, target_shape):
    m = np.load(mask_npy_path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != target_shape:
        m = cv2.resize(m.astype(np.uint8), (target_shape[1], target_shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return (m > 0)

def to_homogeneous(M34):
    if M34.shape == (4,4):
        return M34.astype(np.float32)
    if M34.shape == (3,4):
        return np.vstack([M34.astype(np.float32), [0,0,0,1]])
    raise ValueError(f"Extrinsic must be 3x4 or 4x4, got {M34.shape}")

def save_ply(path, pts, colors=None):
    # Writes either XYZ or XYZRGB; same math, just I/O utility.
    if pts.size == 0:
        print(f"[WARN] empty cloud: {path}"); return
    if colors is None:
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x, y, z in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    else:
        C8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(pts, C8):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# ----------------------------
# Core per-file processing (exact same computation)
# ----------------------------
def process_one(depth_npy, rgb_path, mask_img, mask_npy,
                intrinsic_npy, extrinsic_npy, ply_out,
                f=None, fov_deg=None, voxel_down_m=0.0):
    # Depth
    Z = np.load(depth_npy).astype(np.float32)
    if Z.ndim != 2:
        raise ValueError("depth_npy must be HxW float array.")
    H, W = Z.shape
    print(f"[Depth] {depth_npy} shape={Z.shape}, finite={np.isfinite(Z).sum()}/{Z.size}")

    # Mask (optional)
    mask = None
    if mask_img and os.path.exists(mask_img):
        mask = load_mask_from_image(mask_img, (H, W))
        print(f"[Mask(img)] shape={mask.shape}, true={int(mask.sum())}")
    elif mask_npy and os.path.exists(mask_npy):
        mask = load_mask_from_npy(mask_npy, (H, W))
        print(f"[Mask(npy)] shape={mask.shape}, true={int(mask.sum())}")
    if mask is not None:
        Z = np.where(mask, Z, np.nan)

    # Intrinsics
    if intrinsic_npy and os.path.exists(intrinsic_npy):
        K = np.load(intrinsic_npy).astype(np.float32)
        if K.shape != (3,3):
            raise ValueError(f"Intrinsic file must be 3x3, got {K.shape}")
        print("[Intrinsics] Loaded:", intrinsic_npy, "\n", K)
    else:
        K = build_intrinsics(W, H, f=f, fov_deg=fov_deg)
        print("[Intrinsics] Built from f/fov\n", K)

    # Backproject to CAMERA coords (UNCHANGED math)
    pts_cam, valid_mask = backproject_to_camera(Z, K)
    print(f"[Points@cam] {pts_cam.shape[0]} valid points")

    # Colors (optional)
    colors = extract_colors(rgb_path, target_shape=(H, W), valid_mask_flat=valid_mask.reshape(-1))

    # If extrinsic is provided, transform CAMERA -> WORLD (since extrinsic is WORLD->CAM)
    pts_out = pts_cam
    if extrinsic_npy and os.path.exists(extrinsic_npy):
        E = to_homogeneous(np.load(extrinsic_npy).astype(np.float32))
        E_inv = np.linalg.inv(E)  # CAM->WORLD
        pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1), dtype=np.float32)], axis=1)
        pts_out = (E_inv @ pts_h.T).T[:, :3]
        print("[Extrinsics] Output transformed to WORLD coordinates")

    # Build/Open3D point cloud with optional voxel down
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_out)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if voxel_down_m and voxel_down_m > 0:
        pcd = pcd.voxel_down_sample(voxel_down_m)
        pts_out = np.asarray(pcd.points)
        if colors is not None and len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)

    out_dir = os.path.dirname(ply_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = o3d.io.write_point_cloud(ply_out, pcd,
                                  write_ascii=False, compressed=False, print_progress=True)
    if not ok:
        raise RuntimeError("Failed to write PLY.")
    np.save(os.path.join(out_dir, "points.npy"), pts_out.astype(np.float32))
    if colors is not None:
        np.save(os.path.join(out_dir, "colors.npy"), (colors.astype(np.float32)))
    print(f"[OK] Saved: {ply_out}")

# ----------------------------
# Batch discovery / filtering
# ----------------------------
def find_rel_dirs(root, pattern):
    hits = []
    for p in glob.glob(os.path.join(root, "**", pattern), recursive=True):
        d = os.path.dirname(p)
        rel = os.path.relpath(d, root)
        if rel == ".": rel = ""
        hits.append(rel)
    return sorted(set(hits))

def filter_rel_dirs(all_dirs, obj=None, cams=None):
    if obj and cams:
        wanted = {f"{obj}_{c}" for c in cams}
        return [d for d in all_dirs if d in wanted]
    if obj and not cams:
        pref = f"{obj}_"
        return [d for d in all_dirs if d.startswith(pref)]
    if (not obj) and cams:
        suffix = {f"_{c}" for c in cams}
        return [d for d in all_dirs if any(d.endswith(s) for s in suffix)]
    return all_dirs

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Backproject aligned depth maps to point clouds (single or batch).")
    # SINGLE-FILE arguments (unchanged)
    ap.add_argument("--depth_npy", required=False, help="Depth .npy (HxW float32, in METERS)")
    ap.add_argument("--rgb", required=False, help="Aligned RGB image (for colors)")
    ap.add_argument("--mask_img", required=False, help="Foreground mask image (PNG/JPG). Non-zero => foreground")
    ap.add_argument("--mask_npy", required=False, help="Foreground mask .npy (HxW or HxWx1)")
    ap.add_argument("--ply_out", required=False, help="Output PLY path")
    ap.add_argument("--intrinsic_npy", type=str, default=None,
                    help="Path to 3x3 intrinsic .npy (overrides --f/--fov_deg)")
    ap.add_argument("--extrinsic_npy", type=str, default=None,
                    help="3x4 or 4x4 WORLD->CAM extrinsic .npy (X_cam=R*X_world+t). If set, output in WORLD coords.")
    ap.add_argument("--f", type=float, default=None, help="Focal length in pixels (fx=fy=f) if no intrinsic file")
    ap.add_argument("--fov_deg", type=float, default=None, help="Horizontal FOV in degrees (alt to --f)")
    ap.add_argument("--voxel_down_m", type=float, default=0.0, help="Optional voxel size (meters)")

    # BATCH mode (new)
    ap.add_argument("--aligned_root", type=str, default=None,
                    help="Root containing <obj>_<cam>/<depth_name> for batch")
    ap.add_argument("--parm_root", type=str, default=None,
                    help="Root containing <obj>_<cam>/(intrinsic_name, extrinsic_name)")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Output root; per-view results under <out_root>/<obj>_<cam>/")
    ap.add_argument("--obj", type=str, default=None, help="Filter: object id, e.g. 0004")
    ap.add_argument("--cams", nargs="+", default=None, help="Filter: cameras, e.g. 000 006 011")
    ap.add_argument("--depth_name", default="0_aligned.npy")
    ap.add_argument("--intrinsic_name", default="0_intrinsic.npy")
    ap.add_argument("--extrinsic_name", default="0_extrinsic.npy")
    ap.add_argument("--rgb_root", type=str, default=None)
    ap.add_argument("--rgb_name", default="0.png")
    ap.add_argument("--mask_img_root", type=str, default=None)
    ap.add_argument("--mask_img_name", default="mask.png")
    ap.add_argument("--mask_npy_root", type=str, default=None)
    ap.add_argument("--mask_npy_name", default="0_mask.npy")

    args = ap.parse_args()

    # Decide mode
    single_mode = args.depth_npy is not None

    if single_mode:
        if not args.ply_out:
            raise ValueError("--ply_out is required in single mode")
        process_one(
            depth_npy=args.depth_npy,
            rgb_path=args.rgb,
            mask_img=args.mask_img,
            mask_npy=args.mask_npy,
            intrinsic_npy=args.intrinsic_npy,
            extrinsic_npy=args.extrinsic_npy,
            ply_out=args.ply_out,
            f=args.f, fov_deg=args.fov_deg, voxel_down_m=args.voxel_down_m
        )
        return

    # Batch mode
    if not (args.aligned_root and args.parm_root and args.out_root):
        raise ValueError("Batch mode requires --aligned_root, --parm_root, and --out_root.")

    # discover and filter rel dirs
    rel_dirs = find_rel_dirs(args.aligned_root, args.depth_name)
    rel_dirs = filter_rel_dirs(rel_dirs, obj=args.obj, cams=args.cams)
    print(f"[scan] found {len(rel_dirs)} views")

    if not rel_dirs:
        print("[hint] No matches. Check roots, names, or filters.")
        return

    for rel in rel_dirs:
        depth_path = os.path.join(args.aligned_root, rel, args.depth_name)
        kin_path   = os.path.join(args.parm_root,   rel, args.intrinsic_name)
        ext_path   = os.path.join(args.parm_root,   rel, args.extrinsic_name)
        rgb_path   = (os.path.join(args.rgb_root,   rel, args.rgb_name) if args.rgb_root else None)
        mask_img   = (os.path.join(args.mask_img_root, rel, args.mask_img_name) if args.mask_img_root else None)
        mask_npy   = (os.path.join(args.mask_npy_root, rel, args.mask_npy_name) if args.mask_npy_root else None)

        out_dir = os.path.join(args.out_root, rel)
        ply_out = os.path.join(out_dir, "points.ply")

        try:
            process_one(
                depth_npy=depth_path,
                rgb_path=rgb_path,
                mask_img=mask_img,
                mask_npy=mask_npy,
                intrinsic_npy=kin_path if os.path.isfile(kin_path) else None,
                extrinsic_npy=ext_path if os.path.isfile(ext_path) else None,
                ply_out=ply_out,
                f=args.f, fov_deg=args.fov_deg, voxel_down_m=args.voxel_down_m
            )
        except Exception as e:
            print(f"[{rel}] ERROR: {e}")

if __name__ == "__main__":
    main()
