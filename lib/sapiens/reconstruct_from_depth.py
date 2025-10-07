#!/usr/bin/env python3
import os, argparse, numpy as np, open3d as o3d
import cv2
import open3d as o3d

def to44(M):
    M = np.array(M, dtype=np.float64)
    return M if M.shape == (4,4) else np.vstack([M, [0,0,0,1]])

# def load_depth_masked(depth_path, seg_path):
#     D = np.load(depth_path).astype(np.float32)  # meters (Z-depth), HxW
#     if seg_path and os.path.isfile(seg_path):
#         seg = np.load(seg_path)
#         if seg.ndim == 3: seg = seg[...,0]
#         if seg.shape != D.shape:
#             seg = cv2.resize(seg.astype(np.float32), (D.shape[1], D.shape[0]), interpolation=cv2.INTER_NEAREST)
#         m = seg > 0
#         D = np.where(m, D, 0.0).astype(np.float32)  # Open3D expects zeros for invalid depth
#     else:
#         # replace NaN with zero for Open3D
#         D = np.where(np.isfinite(D) & (D>0), D, 0.0).astype(np.float32)
#     return D

def load_depth_masked(depth_path, seg_path=None, masked_to="zero", mask_thr=127, invert=False,
                      return_debug=False):
    """
    Load depth (meters, HxW) and apply an optional segmentation mask.
    seg_path can be:
      - .npy: HxW or HxWx1, >0 => foreground
      - image: png/jpg; if RGBA use alpha>0; else grayscale>mask_thr => foreground

    masked_to: "zero" or "nan" for masked-out pixels.
    Open3D requires zeros for invalid depth; NaNs are converted to 0 *at integration time*.
    If return_debug=True, also returns the masked depth BEFORE NaN->0 conversion.
    """
    import os, numpy as np, cv2
    D = np.load(depth_path).astype(np.float32)  # meters
    H, W = D.shape

    def _resize_bool(m):
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)

    # build foreground mask m (True=keep)
    m = None
    if seg_path and os.path.isfile(seg_path):
        ext = os.path.splitext(seg_path)[1].lower()
        if ext == ".npy":
            seg = np.load(seg_path)
            if seg.ndim == 3: seg = seg[..., 0]
            m = _resize_bool(seg > 0)
        else:
            img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read mask image: {seg_path}")
            if img.ndim == 3 and img.shape[2] == 4:
                # alpha channel
                m = _resize_bool(img[..., 3] > 0)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                m = _resize_bool(gray.astype(np.uint8) > int(mask_thr))
    # invert if requested
    if m is not None and invert:
        m = ~m

    # apply mask
    if m is not None:
        if masked_to == "zero":
            D_masked = np.where(m, D, 0.0).astype(np.float32)
        else:  # "nan"
            D_masked = np.where(m, D, np.nan).astype(np.float32)
    else:
        # no mask provided: still zero-out invalid depth
        if masked_to == "zero":
            D_masked = np.where(np.isfinite(D) & (D > 0), D, 0.0).astype(np.float32)
        else:
            D_masked = np.where(np.isfinite(D) & (D > 0), D, np.nan).astype(np.float32)

    # Open3D integration safety: it expects zeros for invalid; keep a debug copy if needed
    D_for_o3d = np.where(np.isfinite(D_masked) & (D_masked > 0), D_masked, 0.0).astype(np.float32)

    return (D_for_o3d, D_masked) if return_debug else D_for_o3d



def fuse_tsdf(triples,  masked_to, mask_thr, invert_mask, voxel=0.004, sdf_trunc=0.02, depth_trunc=3.0, ext_mode="auto"):
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for (depth_p, K_p, E_p, seg_p, rgb_p) in triples:
        #D = load_depth_masked(depth_p, seg_p)                      # meters, zeros=invalid
        D = load_depth_masked(depth_p, seg_p, masked_to=masked_to,
                      mask_thr=mask_thr, invert=invert_mask)
        K = np.load(K_p).astype(np.float64)                        # 3x3
        # E = to44(np.load(E_p))                                     # WORLD->CAM
        # T_c2w = np.linalg.inv(E)                                   # CAM->WORLD for Open3D
        E_disk = to44(np.load(E_p))
        if ext_mode == "w2c":
            T_c2w = np.linalg.inv(E_disk)
            mode_used, rmse_note = "w2c(fixed)", ""
        elif ext_mode == "c2w":
            T_c2w = E_disk
            mode_used, rmse_note = "c2w(fixed)", ""
        else:
            # auto: decide per view using a few back-projected points
            T_c2w, chosen, rmseA, rmseB = pick_T_c2w_auto(K, E_disk, D)
            mode_used = f"auto→{chosen}"
            rmse_note = f" (rmseA={rmseA:.2f}px, rmseB={rmseB:.2f}px)"
        print(f"  - pose: {mode_used}{rmse_note}")

        H, W = D.shape

        pin = o3d.camera.PinholeCameraIntrinsic()
        pin.set_intrinsics(W, H, K[0,0], K[1,1], K[0,2], K[1,2])

        depth_o3d = o3d.geometry.Image(D)                          # already in meters
        if rgb_p and os.path.isfile(rgb_p):
            bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (H, W):
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.full((H, W, 3), 127, np.uint8)
        color_o3d = o3d.geometry.Image(rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        vol.integrate(rgbd, pin, T_c2w)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Ensure outward orientation without using invert_normals()
    mesh.orient_triangles()  # make faces consistent first

    tri = np.asarray(mesh.triangles)
    mesh.triangles = o3d.utility.Vector3iVector(tri[:, [0, 2, 1]])  # swap two indices
    mesh.compute_vertex_normals()

    return mesh

def to44(M):
    M = np.array(M, dtype=np.float64)
    return M if M.shape == (4,4) else np.vstack([M, [0,0,0,1]])

def backproject_sample(K, D, n=5000):
    """Sample a few valid pixels, backproject to CAMERA coords as (N,3) and keep their (u,v)."""
    H, W = D.shape
    valid = (D > 0) & np.isfinite(D)
    ys, xs = np.where(valid)
    if ys.size == 0:
        return None, None, None
    idx = np.random.default_rng(0).choice(ys.size, size=min(n, ys.size), replace=False)
    ys, xs = ys[idx].astype(np.float64), xs[idx].astype(np.float64)
    z = D[ys.astype(int), xs.astype(int)].astype(np.float64)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (xs - cx)/fx * z
    Y = (ys - cy)/fy * z
    Pcam = np.stack([X, Y, z], axis=1)    # (N,3)
    return xs, ys, Pcam

def repro_rmse(K, R, t, Xw, u_ref, v_ref):
    """Project Xw with WORLD→CAM [R|t] and return pixel RMSE."""
    Xc = (R @ Xw.T + t).T
    u = (K[0,0]*Xc[:,0] + K[0,2]*Xc[:,2]) / (Xc[:,2] + 1e-12)
    v = (K[1,1]*Xc[:,1] + K[1,2]*Xc[:,2]) / (Xc[:,2] + 1e-12)
    return float(np.mean(np.hypot(u - u_ref, v - v_ref)))

def pick_T_c2w_auto(K, E_w2c, D):
    """
    Given K, a candidate WORLD→CAM extrinsic (from disk), and the depth image D,
    decide whether disk extrinsic is W→C or C→W. Return CAM→WORLD for TSDF.
    """
    # Sample a small set of camera points
    u, v, Xc = backproject_sample(K, D, n=3000)
    if Xc is None:  # no valid depth
        return np.linalg.inv(E_w2c)  # default

    # Case A: assume disk is W→C
    T_c2w_A = np.linalg.inv(E_w2c)
    Xw_A = (T_c2w_A[:3,:3] @ Xc.T + T_c2w_A[:3,3:4]).T
    rmse_A = repro_rmse(K, E_w2c[:3,:3], E_w2c[:3,3:4], Xw_A, u, v)

    # Case B: assume disk is C→W
    T_c2w_B = E_w2c
    # For reprojection we need a WORLD→CAM; inverse of C→W is W→C
    E_w2c_B = np.linalg.inv(T_c2w_B)
    Xw_B = (T_c2w_B[:3,:3] @ Xc.T + T_c2w_B[:3,3:4]).T
    rmse_B = repro_rmse(K, E_w2c_B[:3,:3], E_w2c_B[:3,3:4], Xw_B, u, v)

    return (T_c2w_A if rmse_A <= rmse_B else T_c2w_B), ("w2c" if rmse_A <= rmse_B else "c2w"), rmse_A, rmse_B


def main():
    ap = argparse.ArgumentParser(description="Fuse 3 masked aligned depths into a mesh (TSDF).")
    ap.add_argument("--aligned_root", required=True)   # <obj>_<cam>/0_aligned.npy
    ap.add_argument("--parm_root",    required=True)   # <obj>_<cam>/0_intrinsic.npy, 0_extrinsic.npy
    ap.add_argument("--seg_root",     default=None)    # <obj>_<cam>/seg.npy  (mask>0 kept). Optional.
    ap.add_argument("--rgb_root",     default=None)    # optional <obj>_<cam>/0.jpg (for colors)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--cams", nargs=3, required=True)
    ap.add_argument("--depth_name", default="0_aligned.npy")
    ap.add_argument("--K_name",     default="0_intrinsic.npy")
    ap.add_argument("--E_name",     default="0_extrinsic.npy")
    ap.add_argument("--seg_name",   default="0_seg.npy")   # change if your mask filename differs
    ap.add_argument("--rgb_name",   default="0.jpg")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--sdf_trunc", type=float, default=0.02)
    ap.add_argument("--depth_trunc", type=float, default=3.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ext_mode", choices=["w2c","c2w","auto"], default="auto",
                help="Meaning of the extrinsic on disk. TSDF needs CAM→WORLD; 'auto' picks per view.")
    ap.add_argument("--masked_to", choices=["zero","nan"], default="zero",
                help="Set masked-out depth to 0 (recommended for TSDF) or NaN (for debugging).")
    ap.add_argument("--mask_thr", type=int, default=127,
                    help="Binarization threshold for image masks (0-255). >thr => foreground.")
    ap.add_argument("--invert_mask", action="store_true",
                    help="Invert foreground/background after thresholding.")


    args = ap.parse_args()

    triples = []
    for cam in args.cams:
        rel = f"{args.obj}_{cam}"
        dpth = os.path.join(args.aligned_root, rel, args.depth_name)
        kin  = os.path.join(args.parm_root,    rel, args.K_name)
        ext  = os.path.join(args.parm_root,    rel, args.E_name)
        seg  = os.path.join(args.seg_root,     rel, args.seg_name) if args.seg_root else None
        rgb  = os.path.join(args.rgb_root,     rel, args.rgb_name) if args.rgb_root else None
        if not (os.path.isfile(dpth) and os.path.isfile(kin) and os.path.isfile(ext)):
            print(f"[{rel}] missing depth/K/E → skip"); continue
        triples.append((dpth, kin, ext, seg, rgb))

    if not triples:
        raise SystemExit("No valid views.")

    #mesh = fuse_tsdf(triples, voxel=args.voxel, sdf_trunc=args.sdf_trunc, depth_trunc=args.depth_trunc)
    # mesh = fuse_tsdf(triples, mvoxel=args.voxel, sdf_trunc=args.sdf_trunc,
    #              depth_trunc=args.depth_trunc, ext_mode=args.ext_mode)
    mesh = fuse_tsdf(
        triples,
        masked_to=args.masked_to,
        mask_thr=args.mask_thr,
        invert_mask=args.invert_mask,
        voxel=args.voxel,                 # <- was mvoxel
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        ext_mode=args.ext_mode
    )



    os.makedirs(args.out_dir, exist_ok=True)
    mesh_path = os.path.join(args.out_dir, f"{args.obj}_tsdf.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"[OK] Mesh → {mesh_path}")

    # optional: also dump fused point cloud (samples from mesh vertices)
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    np.save(os.path.join(args.out_dir, f"{args.obj}_fused_points.npy"), pts)
    print(f"[OK] Fused points → {os.path.join(args.out_dir, f'{args.obj}_fused_points.npy')}")

if __name__ == "__main__":
    main()
