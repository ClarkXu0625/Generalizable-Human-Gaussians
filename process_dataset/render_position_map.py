'''
MIT License

Copyright (c) 2024 Youngjoong Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import sys
import math
import trimesh
import os
import pdb

def find_3d_vertices_for_uv(faces):
    uv_to_vertices = {}

    vertices = []
    uvs = []

    for face in faces:
        for idx in range(3):

            vertex_index, uv_index = face[idx]

            if uv_index in uv_to_vertices:
                uv_to_vertices[uv_index].add(vertex_index)
            else:
                uv_to_vertices[uv_index] = set()
                uv_to_vertices[uv_index].add(vertex_index)

    return uv_to_vertices

def find_3d_vertices_for_uv_torch(faces: torch.Tensor) -> dict[int, set[int]]:
    """
    Args:
        faces: [F, 3, 2] tensor of (vertex_idx, uv_idx) per corner
    Returns:
        Dictionary mapping uv_index → set(vertex_index)
    """
    uv_to_vertices = {}

    flat = faces.view(-1, 2)  # [F*3, 2]
    vert_idx = flat[:, 0].tolist()
    uv_idx = flat[:, 1].tolist()

    for v_idx, u_idx in zip(vert_idx, uv_idx):
        if u_idx not in uv_to_vertices:
            uv_to_vertices[u_idx] = set()
        uv_to_vertices[u_idx].add(v_idx)

    return uv_to_vertices

def load_obj(file_path):

    vertices = []
    faces = []
    uvs = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == 'v':

                x, y, z = map(float, tokens[1:4])
                vertices.append((x, y, z))

            elif tokens[0] == 'vt':

                u, v = map(float, tokens[1:3])
                uvs.append((u, v))

            elif tokens[0] == 'f':

                face = []
                for token in tokens[1:]:
                    vertex_info = token.split('/')
                    vertex_index = int(vertex_info[0]) - 1
                    uv_index = int(vertex_info[1]) - 1 if len(
                        vertex_info) > 1 else None
                    face.append((vertex_index, uv_index))
                faces.append(face)

    return vertices, faces, uvs


num_angles = 16
phase = 'train'  # 'val' #'train'
data_root = 'datasets/THuman/{}'.format(phase) # 'datasets/THuman/THuman2.0_Release'.format(phase) #
calib_dir = os.path.join(data_root, 'parm')
depth_dir = os.path.join(data_root, 'depth')
human_list = set()

for dir_name in os.listdir(os.path.join(data_root, 'img')):
    subject_name = dir_name.split('_')[0]
    human_list.add(subject_name)
human_list = list(human_list)
human_list.sort()

resolution = 1024

# image plane shape
image_height = 1024
image_width = 1024

glctx = dr.RasterizeCudaContext()

smplx_fp = "datasets/THuman/smplx_uv.obj"

# Load SMPLX UV OBJ
vertices_tpose, faces, uvs = load_obj(smplx_fp)

vertices_tpose = torch.tensor(vertices_tpose, dtype=torch.float32, device='cuda')  # [N, 3]
faces = torch.tensor(faces, dtype=torch.int32, device='cuda')                      # [M, 3, 2]
uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')                        # [N_uv, 2]

#uv_pts_mapping = find_3d_vertices_for_uv(faces.cpu().numpy())
uv_pts_mapping = find_3d_vertices_for_uv_torch(faces)

# Build UV-space vertex positions in clip space
pos = 2.0 * uvs - 1.0
zeros = torch.zeros_like(pos[:, :1])
ones = torch.ones_like(pos[:, :1])
final_pos = torch.cat([pos, zeros, ones], dim=-1).unsqueeze(0)  # [1, N_uv, 4]

pos_uv = final_pos.contiguous()
tri_uv = faces[:, :, 1]  # use uv indices
#rast_uv_space, _ = dr.rasterize(glctx, pos_uv, tri_uv, resolution=[resolution, resolution])
rast_uv_space, _ = dr.rasterize(glctx, pos_uv.contiguous(), tri_uv.contiguous(), resolution=[resolution, resolution])

face_id_raw = rast_uv_space[..., 3:]
face_id = face_id_raw[0]

# Per human
for scale_idx in range(5):
    scale = scale_idx * 0.01
    position_dir = os.path.join(data_root, f'position_map_uv_space_outer_shell_{scale_idx}' if scale_idx > 0 else 'position_map_uv_space')
    os.makedirs(position_dir, exist_ok=True)

    img_dir = os.path.join(position_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    for human_idx, human in enumerate(human_list):
        obj_file_path = os.path.join(data_root, 'smplx_obj', f'{human}.obj')

        # Load and convert to torch
        vertices_np, _, _ = load_obj(obj_file_path)
        vertices = torch.tensor(vertices_np, dtype=torch.float32, device='cuda')  # [N, 3]
        normals = torch.tensor(trimesh.load(obj_file_path, process=False).vertex_normals, dtype=torch.float32, device='cuda')  # [N, 3]
        vertices = vertices + scale * normals

        # Attribute for each UV point: gather from mapped vertices
        attr_list = [vertices[list(uv_pts_mapping[uv_idx])].mean(dim=0) for uv_idx in range(uvs.shape[0])]
        attr = torch.stack(attr_list, dim=0).unsqueeze(0)  # [1, N_uv, 3]

        # Rasterize to UV-space
        out2, _ = dr.interpolate(attr.contiguous(), rast_uv_space.contiguous(), tri_uv.contiguous())  # [1, H, W, 3]

        bg_indices = torch.nonzero(face_id <= 0)
        out2[:, bg_indices[:, 0], bg_indices[:, 1], 0] = 0
        out2[:, bg_indices[:, 0], bg_indices[:, 1], 1] = 0
        out2[:, bg_indices[:, 0], bg_indices[:, 1], 2] = 0

        # out3: [H, W, 3], already on GPU
        out3 = out2[0].flip(0)  # Flip vertically to match image orientation

        # Save .npy (optional)
        position_map_name = os.path.join(position_dir, f'{human}_{resolution}.npy')
        np.save(position_map_name, out3.cpu().numpy().astype(np.float32))

        # Pure Torch visualization
        flat = out3.view(-1, 3)
        min_vals, _ = flat.min(dim=0, keepdim=True)
        max_vals, _ = flat.max(dim=0, keepdim=True)
        norm_pos = (out3 - min_vals) / (max_vals - min_vals + 1e-8)
        norm_pos = norm_pos.clamp(0, 1)
        vis_rgb = (norm_pos * 255).to(torch.uint8)
        
        png_filename = os.path.basename(position_map_name).replace('.npy', '.png')
        png_path = os.path.join(img_dir, png_filename)

        imageio.imwrite(png_path, vis_rgb.cpu().numpy())

        print(f"Saved position map visualization: {png_path}")

        #pdb.set_trace()
