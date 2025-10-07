import argparse, os
import numpy as np
import cv2
import open3d as o3d

def build_intrinsics(W, H, f=None, fov_deg=None):
    """
    Create K with principal point at center and fx=fy=f.
    If f is None, derive from horizontal FOV (default 60°).
    """
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
    """
    Backproject (u,v,Z) -> (X,Y,Z) in camera coords (meters).
    X right, Y down, Z forward.
    """
    H, W = Z_m.shape
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)          # HxW

    Zf = Z_m.reshape(-1)
    uf = u.reshape(-1)
    vf = v.reshape(-1)

    good = np.isfinite(Zf) & (Zf > 0)
    Zf = Zf[good]
    uf = uf[good]
    vf = vf[good]

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
    """
    Load a mask from an image (PNG/JPG). Any non-zero pixel => foreground.
    If RGBA, use alpha channel.
    """
    img = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read mask image: {mask_img_path}")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        m = alpha > 0
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_npy", required=True, help="Depth .npy (HxW float32, in METERS)")
    ap.add_argument("--rgb", required=False, help="Aligned RGB image (for colors)")

    # prefer image mask; npy also supported
    ap.add_argument("--mask_img", required=False, help="Foreground mask image (PNG/JPG). Non-zero => foreground")
    ap.add_argument("--mask_npy", required=False, help="Foreground mask .npy (HxW or HxWx1)")

    ap.add_argument("--ply_out", required=True, help="Output PLY path")

    # intrinsics
    ap.add_argument("--f", type=float, default=None, help="Focal length in pixels (fx=fy=f)")
    ap.add_argument("--fov_deg", type=float, default=None, help="Horizontal FOV in degrees (alt to --f)")
    ap.add_argument("--intrinsic_npy", type=str, default=None,
                    help="Path to .npy file containing a 3x3 intrinsic matrix (overrides --f and --fov_deg)")

    # optional extrinsic (WORLD->CAM). If given, output points transformed to WORLD.
    ap.add_argument("--extrinsic_npy", type=str, default=None,
                    help="3x4 or 4x4 WORLD->CAM extrinsic .npy (X_cam=R*X_world+t). If set, output in WORLD coords.")

    # cloud options
    ap.add_argument("--voxel_down_m", type=float, default=0.0, help="Optional voxel size (meters)")
    args = ap.parse_args()

    # Depth (already meters)
    Z = np.load(args.depth_npy).astype(np.float32)
    if Z.ndim != 2:
        raise ValueError("depth_npy must be HxW float array.")
    H, W = Z.shape
    print(f"[Depth] shape={Z.shape}, finite={np.isfinite(Z).sum()}/{Z.size}")

    # Mask
    mask = None
    if args.mask_img and os.path.exists(args.mask_img):
        mask = load_mask_from_image(args.mask_img, target_shape=(H, W))
        print(f"[Mask(img)] shape={mask.shape}, true={int(mask.sum())}")
    elif args.mask_npy and os.path.exists(args.mask_npy):
        mask = load_mask_from_npy(args.mask_npy, target_shape=(H, W))
        print(f"[Mask(npy)] shape={mask.shape}, true={int(mask.sum())}")

    if mask is not None:
        # zero-out invalid pixels (helps later)
        Z = np.where(mask, Z, np.nan)

    # Intrinsics
    if args.intrinsic_npy is not None and os.path.exists(args.intrinsic_npy):
        K = np.load(args.intrinsic_npy).astype(np.float32)
        if K.shape != (3,3):
            raise ValueError(f"Intrinsic file must be 3x3, got {K.shape}")
        print("[Intrinsics] Loaded from file:", args.intrinsic_npy)
        print(K)
    else:
        K = build_intrinsics(W, H, f=args.f, fov_deg=args.fov_deg)
        print("[Intrinsics] Built from f/fov")
        print(K)

    # Backproject to CAMERA coords
    pts_cam, valid_mask = backproject_to_camera(Z, K)
    print(f"[Points@cam] {pts_cam.shape[0]} valid points")

    # Colors
    colors = extract_colors(args.rgb, target_shape=(H, W), valid_mask_flat=valid_mask.reshape(-1))

    # If extrinsic is provided, transform CAMERA -> WORLD (since extrinsic is WORLD->CAM)
    pts_out = pts_cam
    if args.extrinsic_npy and os.path.exists(args.extrinsic_npy):
        E = to_homogeneous(np.load(args.extrinsic_npy).astype(np.float32))
        E_inv = np.linalg.inv(E)  # CAM->WORLD
        pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1), dtype=np.float32)], axis=1)  # Nx4
        pts_out = (E_inv @ pts_h.T).T[:, :3]
        print("[Extrinsics] Output transformed to WORLD coordinates")

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_out)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if args.voxel_down_m and args.voxel_down_m > 0:
        pcd = pcd.voxel_down_sample(args.voxel_down_m)

    out_dir = os.path.dirname(args.ply_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = o3d.io.write_point_cloud(args.ply_out, pcd,
                                  write_ascii=False, compressed=False, print_progress=True)
    if not ok:
        raise RuntimeError("Failed to write PLY.")
    print(f"[OK] Saved: {args.ply_out}")

if __name__ == "__main__":
    main()
