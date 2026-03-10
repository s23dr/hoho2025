"""
Copyright [2022] [Paul-Edouard Sarlin and Philipp Lindenberger]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

3D visualization based on plotly.
Works for a small number of points and cameras, might be slow otherwise.

1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums, or both as a pycolmap.Reconstruction

Written by Paul-Edouard Sarlin and Philipp Lindenberger.
Slightly modified by Dmytro Mishkin
"""
from typing import Optional
import numpy as np
import pycolmap
import plotly.graph_objects as go
from hoho2025.color_mappings import edge_color_mapping, EDGE_CLASSES_BY_ID

def to_homogeneous(points):
    pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)

### Plotting functions

def init_figure(height: int = 800, reverse_gravity: bool = False) -> go.Figure:
    """Initialize a 3D figure.

    Args:
        height: Figure height in pixels.
        reverse_gravity: Set to ``True`` for the **2025** dataset, whose
            coordinate frame has Y pointing *down* (the original SketchUp /
            COLMAP convention before the 2026 re-orientation).  When ``False``
            (default, for the **2026** dataset) the viewer is set up for a
            standard Y-up world so that the roof wireframe appears right-side up.
    """
    fig = go.FigureWidget()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    if reverse_gravity:
        # 2025 data: Y points down — look from below with Y-down up-vector.
        scene_camera = dict(
            eye=dict(x=0., y=-.1, z=-2.),
            up=dict(x=0, y=-1., z=0),
            projection=dict(type="orthographic"))
    else:
        # 2026 data: Y points up — standard bird's-eye view.
        scene_camera = dict(
            eye=dict(x=0., y=1.5, z=-3.),
            up=dict(x=0, y=1., z=0),
            projection=dict(type="orthographic"))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=scene_camera,
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode='data',
            dragmode='orbit',
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.1
        ),
    )
    return fig


def plot_lines_3d(
        fig: go.Figure,
        pts: np.ndarray,
        color: str = 'rgba(255, 255, 255, 1)',
        ps: int = 2,
        colorscale: Optional[str] = None,
        name: Optional[str] = None):
    """Plot a set of 3D points."""
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]
    if isinstance(color, list):
        traces = [go.Scatter3d(x=x1, y=y1, z=z1,
                            mode='lines',
                            line=dict(color=f"rgb{c}", width=ps)) for x1, y1, z1, c in zip(x,y,z,color)]
    else:
        traces = [go.Scatter3d(x=x1, y=y1, z=z1,
                        mode='lines',
                        line=dict(color=color, width=ps)) for x1, y1, z1 in zip(x,y,z)]
    for t in traces:
        fig.add_trace(t)
    fig.update_traces(showlegend=False)


def plot_points(
        fig: go.Figure,
        pts: np.ndarray,
        color: str = 'rgba(255, 0, 0, 1)',
        ps: int = 2,
        colorscale: Optional[str] = None,
        name: Optional[str] = None):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x, y=y, z=z, mode='markers', name=name, legendgroup=name,
        marker=dict(
            size=ps, color=color, line_width=0.0, colorscale=colorscale))
    fig.add_trace(tr)

def plot_camera(
        fig: go.Figure,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        legendgroup: Optional[str] = None,
        size: float = 1.0):
    """Plot a camera frustum from pose and intrinsic matrix. R and t are 
    world_to_camera transformation"""
    R = np.array(R)
    t = np.array(t).reshape(3)
    K = np.array(K)
    W, H = K[0, 2]*2, K[1, 2]*2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t

    x, y, z = corners.T
    rect = go.Scatter3d(
        x=x, y=y, z=z, line=dict(color=color), legendgroup=legendgroup,
        name=name, marker=dict(size=0.0001), showlegend=False)
    fig.add_trace(rect)

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    pyramid = go.Mesh3d(
        x=x, y=y, z=z, color=color, i=i, j=j, k=k,
        legendgroup=legendgroup, name=name, showlegend=False)
    fig.add_trace(pyramid)
    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([
        vertices[i] for i in triangles.reshape(-1)
    ])

    x, y, z = tri_points.T
    pyramid = go.Scatter3d(
        x=x, y=y, z=z, mode='lines', legendgroup=legendgroup,
        name=name, line=dict(color=color, width=1), showlegend=False)
    fig.add_trace(pyramid)


def plot_camera_colmap(
        fig: go.Figure,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
        name: Optional[str] = None,
        **kwargs):
    """Plot a camera frustum from PyCOLMAP objects"""
    # Use camera intrinsics method if available, otherwise fallback to params
    intr = camera.calibration_matrix()
    if intr[0][0] > 5000:
        print("Bad camera")
        return
    world_t_camera = image.cam_from_world.inverse()
    plot_camera(
        fig,
        world_t_camera.rotation.matrix(),  # Use rotation matrix method (World-to-Camera)
        world_t_camera.translation,  # Use camera center in world coordinates
        intr,
        name=name or str(image.name),
        **kwargs)


def plot_cameras(
        fig: go.Figure,
        reconstruction: pycolmap.Reconstruction, # Added type hint
        **kwargs):
    """Plot a camera as a cone with camera frustum."""
    # Iterate over reconstruction.images
    for image_id, image in reconstruction.images.items():
        # Access camera using reconstruction.cameras
        plot_camera_colmap(
            fig, image, reconstruction.cameras[image.camera_id], **kwargs)


def plot_reconstruction(
        fig: go.Figure,
        rec: pycolmap.Reconstruction, # Added type hint
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        points: bool = True,
        cameras: bool = True,
        cs: float = 1.0,
        single_color_points=False,
        camera_color='rgba(0, 255, 0, 0.5)',
        crop_outliers: bool = False):
    # rec is a pycolmap.Reconstruction object
    # Filter outliers
    xyzs = []
    rgbs = []
    # Iterate over rec.points3D
    for k, p3D in rec.points3D.items():
        xyzs.append(p3D.xyz)
        rgbs.append(p3D.color)
    
    xyzs = np.array(xyzs)
    rgbs = np.array(rgbs)
    
    # Crop outliers if requested
    if crop_outliers and len(xyzs) > 0:
        # Calculate distances from origin
        distances = np.linalg.norm(xyzs, axis=1)
        # Find threshold at 98th percentile (removing 2% furthest points)
        threshold = np.percentile(distances, 98)
        # Filter points
        mask = distances <= threshold
        xyzs = xyzs[mask]
        rgbs = rgbs[mask]
        print(f"Cropped outliers: removed {np.sum(~mask)} out of {len(mask)} points ({np.sum(~mask)/len(mask)*100:.2f}%)")

    if points and len(xyzs) > 0:
        plot_points(fig, xyzs, color=color if single_color_points else rgbs, ps=1, name=name)
    if cameras:
        plot_cameras(fig, rec, color=camera_color, legendgroup=name, size=cs)

def plot_wireframe(
        fig: go.Figure,
        vertices: np.ndarray,
        edges: np.ndarray,
        classifications: np.ndarray = None,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        **kwargs):
    """Plot a wireframe with per-edge semantic colors."""
    gt_vertices = np.array(vertices)
    gt_connections = np.array(edges)
    if gt_vertices is not None:
        img_color2 = [color for _ in range(len(gt_vertices))]
        plot_points(fig, gt_vertices, color = img_color2, ps = 10)  
        if gt_connections is not None:
            gt_lines = []
            for c in gt_connections:
                v1 = gt_vertices[c[0]]
                v2 = gt_vertices[c[1]]
                gt_lines.append(np.stack([v1, v2], axis=0))
            if classifications is not None and len(classifications) == len(gt_lines):
                line_colors = []
                for c in classifications:
                    line_colors.append(edge_color_mapping[EDGE_CLASSES_BY_ID[c]])
                plot_lines_3d(fig, np.array(gt_lines), line_colors, ps=4)  
            else:
                plot_lines_3d(fig, np.array(gt_lines), color, ps=4)  


def plot_bpo_cameras_from_entry(
        fig: go.Figure,
        entry: dict,
        idx: Optional[int] = None,
        color: str = 'rgb(255, 128, 0)',
        size: float = 1.0):
    """Plot BPO (DAE) camera frustums for a dataset entry.

    Cameras flagged as ``pose_only_in_colmap=True`` are skipped because their
    K / R / t are all zeros and would cause a singular-matrix error.

    Supports both the 2025 format (``colmap_binary``) and the 2026 format
    (``colmap``, ``pose_only_in_colmap`` per-camera flag).
    """
    pose_only_flags = entry.get('pose_only_in_colmap', [])

    def cam2world_to_world2cam(R, t):
        # Rᵀ(p_cam − t) → R_w2c = Rᵀ, t_w2c = −Rᵀ t
        R = np.array(R, dtype=np.float64)
        t = np.array(t, dtype=np.float64).reshape(3)
        R_w2c = R.T
        t_w2c = -R_w2c @ t
        return R_w2c, t_w2c

    for i in range(len(entry['R'])):
        if idx is not None and i != idx:
            continue
        # Skip cameras that exist only in COLMAP (zero K/R/t).
        if i < len(pose_only_flags) and pose_only_flags[i]:
            continue
        K = np.array(entry['K'][i])
        # Guard against all-zero K from old loaders that may not set pose_only flags.
        if np.allclose(K, 0.0):
            continue
        R = np.array(entry['R'][i])
        t = np.array(entry['t'][i])
        R_w2c, t_w2c = cam2world_to_world2cam(R, t)
        plot_camera(fig, R_w2c, t_w2c, K, color=color, size=size)


# ---------------------------------------------------------------------------
# Depth + segmentation unprojection helpers
# ---------------------------------------------------------------------------

def _open_image_field(img_field):
    """Convert an HF Image() field value (PIL Image, bytes-dict, or raw bytes) to PIL Image."""
    import io as _io
    from PIL import Image as PILImage
    if img_field is None:
        return None
    if isinstance(img_field, PILImage.Image):
        return img_field
    raw = None
    if isinstance(img_field, dict) and "bytes" in img_field:
        raw = img_field["bytes"]
    elif isinstance(img_field, (bytes, bytearray)):
        raw = img_field
    if raw is None:
        return None
    try:
        return PILImage.open(_io.BytesIO(raw))
    except Exception:
        return None


def _resolve_skip_colors(skip_classes):
    """
    Return a uint8 array of shape (K, 3) with the ADE20k RGB colours for
    *skip_classes*, or None if no classes could be resolved.

    Matching rules (case-insensitive):
      1. Exact key match      – 'sky'    → 'sky'
      2. Semicolon-part match – 'window' → 'windowpane;window'
    Unknown names are silently ignored.
    """
    from hoho2025.color_mappings import ade20k_color_mapping
    colors = []
    for cls in skip_classes:
        cls_lower = cls.lower()
        if cls_lower in ade20k_color_mapping:
            colors.append(ade20k_color_mapping[cls_lower])
        else:
            for key, rgb in ade20k_color_mapping.items():
                if cls_lower in [p.strip() for p in key.split(';')]:
                    colors.append(rgb)
                    break
    return np.array(colors, dtype=np.uint8) if colors else None  # (K, 3) or None


def _unproject_depth(depth_pil, ade_rgb, K_np, R_np, t_np,
                     target_size, depth_scale, max_depth, skip_colors_arr):
    """
    Shared unprojection core.  Returns (pts_world, r_ch, g_ch, b_ch) or None.

    K_np  — 3×3 intrinsics (modified in-place to match target_size).
    R_np  — 3×3 cam_from_world rotation.
    t_np  — (3,) cam_from_world translation.
    """
    from PIL import Image as PILImage

    W_t, H_t = target_size
    W_d, H_d = depth_pil.size  # PIL (width, height)

    # Rescale K from its native resolution (inferred from cx) to depth image size,
    # then again to target_size.
    w_K = K_np[0, 2] * 2.0
    h_K = K_np[1, 2] * 2.0
    if w_K > 0 and h_K > 0:
        K_np[0, 0] *= W_d / w_K;  K_np[1, 1] *= H_d / h_K
        K_np[0, 2] *= W_d / w_K;  K_np[1, 2] *= H_d / h_K
    K_np[0, 0] *= W_t / W_d;  K_np[1, 1] *= H_t / H_d
    K_np[0, 2] *= W_t / W_d;  K_np[1, 2] *= H_t / H_d

    depth_arr = np.array(depth_pil, dtype=np.float32)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]
    depth_np = np.array(
        PILImage.fromarray(depth_arr, mode='F').resize((W_t, H_t), PILImage.NEAREST),
        dtype=np.float32) * depth_scale

    if ade_rgb is not None:
        ade_s = np.array(
            PILImage.fromarray(ade_rgb).resize((W_t, H_t), PILImage.NEAREST),
            dtype=np.uint8)
    else:
        ade_s = np.full((H_t, W_t, 3), 180, dtype=np.uint8)

    valid = (depth_np > 0) & (depth_np < max_depth)
    if skip_colors_arr is not None:
        class_mask = np.any(
            np.all(ade_s[:, :, None, :] == skip_colors_arr[None, None, :, :], axis=-1),
            axis=-1)
        valid = valid & ~class_mask
    if not valid.any():
        return None

    u_grid, v_grid = np.meshgrid(np.arange(W_t, dtype=np.float64),
                                  np.arange(H_t, dtype=np.float64))
    pix_h   = np.stack([u_grid[valid], v_grid[valid], np.ones(valid.sum())], axis=1)
    pts_cam = (pix_h @ np.linalg.inv(K_np).T) * depth_np[valid, None].astype(np.float64)
    pts_world = (pts_cam - t_np.reshape(3)) @ R_np  # Rᵀ(p_cam − t)

    return pts_world, ade_s[:, :, 0][valid], ade_s[:, :, 1][valid], ade_s[:, :, 2][valid]


def _load_colmap_from_entry(entry):
    """
    Unzip and parse the COLMAP reconstruction stored in entry['colmap'] or
    entry['colmap_binary'] (old 2025 key name).

    Returns a pycolmap.Reconstruction, or None if the field is absent/invalid.
    """
    import io as _io
    import zipfile
    import tempfile

    colmap_data = entry.get("colmap") or entry.get("colmap_binary")
    if colmap_data is None:
        return None
    if isinstance(colmap_data, list):
        colmap_data = bytes(colmap_data)
    if not isinstance(colmap_data, (bytes, bytearray)) or len(colmap_data) == 0:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(_io.BytesIO(colmap_data), "r") as zf:
                zf.extractall(tmpdir)
            rec = pycolmap.Reconstruction(tmpdir)
            return rec
    except Exception as e:
        print(f"Warning: could not load colmap from entry: {e}")
        return None


def plot_depth_and_segmentation_in_3d(
        fig: go.Figure,
        entry: dict,
        idx: Optional[int] = None,
        target_size: tuple = (128, 96),
        depth_scale: float = 0.001,
        max_depth: float = 64.0,
        skip_classes: Optional[list] = None,
        point_size: int = 2):
    """Unproject depth maps coloured with ADE20k segmentation into a 3D scatter.

    Uses the BPO camera parameters (entry['K'], entry['R'], entry['t']).
    Cameras flagged as ``pose_only_in_colmap`` are skipped automatically.

    In the 2026 format, depth/ade lists may be shorter than the full camera list
    because pose-only cameras have no depth/image files.  This function correctly
    matches each depth entry to its corresponding camera by building a positional
    mapping over non-pose-only cameras.

    Args:
        fig: Plotly figure created by init_figure().
        entry: Dataset entry dict (2025 or 2026 format).
        idx: If set, only process the depth image at this position in the
             depth list (i.e. among non-pose-only cameras).
        target_size: (width, height) to downscale before unprojection.
        depth_scale: Multiply raw pixel values by this to get metres.
        max_depth: Discard pixels deeper than this (metres).
        skip_classes: ADE20k class names to exclude (e.g. ['sky', 'tree']).
        point_size: Plotly marker size.
    """
    skip_colors_arr = _resolve_skip_colors(skip_classes) if skip_classes else None

    depths    = entry.get("depth", []) or []
    aes       = entry.get("ade",   []) or []
    Ks        = entry.get("K", [])
    Rs        = entry.get("R", [])
    ts        = entry.get("t", [])
    pose_only = entry.get("pose_only_in_colmap", [])
    image_ids = entry.get("image_ids", [])

    # Build the list of camera indices (into K/R/t) that actually have depth/ade.
    # In 2026 format, pose-only cameras are interspersed in K/R/t but absent from
    # depth/ade lists, so we cannot use a shared positional counter.
    non_po_cam_indices = [
        i for i in range(len(Ks))
        if not (i < len(pose_only) and pose_only[i]) and not np.allclose(Ks[i], 0.0)
    ]

    for depth_pos, cam_idx in enumerate(non_po_cam_indices):
        if idx is not None and depth_pos != idx:
            continue
        if depth_pos >= len(depths):
            break

        depth_field = depths[depth_pos]
        ade_field   = aes[depth_pos] if depth_pos < len(aes) else None

        depth_pil = _open_image_field(depth_field)
        ade_pil   = _open_image_field(ade_field)
        if depth_pil is None:
            continue

        K_np = np.array(Ks[cam_idx], dtype=np.float64).copy()
        R_np = np.array(Rs[cam_idx], dtype=np.float64)
        t_np = np.array(ts[cam_idx], dtype=np.float64).reshape(3)

        ade_rgb = (np.array(ade_pil.convert("RGB"), dtype=np.uint8)
                   if ade_pil is not None else None)

        result = _unproject_depth(
            depth_pil, ade_rgb,
            K_np=K_np, R_np=R_np, t_np=t_np,
            target_size=target_size,
            depth_scale=depth_scale,
            max_depth=max_depth,
            skip_colors_arr=skip_colors_arr,
        )
        if result is None:
            continue
        pts_world, r_ch, g_ch, b_ch = result

        colors = [f"rgb({r},{g},{b})" for r, g, b in zip(r_ch, g_ch, b_ch)]
        label  = image_ids[cam_idx] if cam_idx < len(image_ids) else str(cam_idx)
        fig.add_trace(go.Scatter3d(
            x=pts_world[:, 0],
            y=pts_world[:, 1],
            z=pts_world[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=colors, line_width=0),
            name=f"depth_{label}",
            showlegend=False,
        ))


def plot_depth_and_segmentation_in_3d_colmap(
        fig: go.Figure,
        entry: dict,
        idx: Optional[int] = None,
        target_size: tuple = (128, 96),
        depth_scale: float = 0.001,
        max_depth: float = 64.0,
        skip_classes: Optional[list] = None,
        point_size: int = 2):
    """Unproject depth maps into 3D using camera poses from the stored COLMAP
    reconstruction (entry['colmap'] or entry['colmap_binary']).

    Unlike :func:`plot_depth_and_segmentation_in_3d`, this variant reads camera
    parameters directly from the COLMAP reconstruction.  This means all cameras
    registered in COLMAP are available, including those flagged as
    ``pose_only_in_colmap``.

    Depth images are matched to COLMAP cameras by ``image_id`` (the same
    lexicographic order used by ds_loader_2026.py for non-pose-only cameras).

    Args:
        fig: Plotly figure created by :func:`init_figure`.
        entry: Dataset entry with keys ``colmap``/``colmap_binary``, ``depth``,
               ``ade``, ``image_ids``, ``pose_only_in_colmap``.
        idx: If set, only process the depth image at this position in the
             depth list (i.e. among non-pose-only cameras).
        target_size: ``(width, height)`` for downscaling before unprojection.
        depth_scale: Scale factor to convert raw pixel values to metres.
        max_depth: Discard pixels whose depth exceeds this value.
        skip_classes: ADE20k class names to exclude (e.g. ``['sky', 'tree']``).
        point_size: Plotly marker size.
    """
    from PIL import Image as PILImage

    skip_colors_arr = _resolve_skip_colors(skip_classes) if skip_classes else None

    rec = _load_colmap_from_entry(entry)
    if rec is None:
        print("plot_depth_and_segmentation_in_3d_colmap: no colmap in entry")
        return

    depths    = entry.get("depth", []) or []
    aes       = entry.get("ade",   []) or []
    image_ids = entry.get("image_ids", [])
    pose_only = entry.get("pose_only_in_colmap", [])

    # Build img_id → (K, R, t) from the COLMAP reconstruction.
    # Image names may be raw ("image_{img_id}_order_{order_id}.jpg") or
    # anonymised hashes ("{img_id}.jpg") depending on the dataset version.
    colmap_cam_map = {}
    for _, img in rec.images.items():
        parts = img.name.split('_')
        img_id = parts[1] if len(parts) >= 2 else img.name.split('.')[0]
        cam    = rec.cameras[img.camera_id]
        K_c    = cam.calibration_matrix()
        R_c    = img.cam_from_world.rotation.matrix()
        t_c    = img.cam_from_world.translation
        colmap_cam_map[img_id] = (K_c, R_c, t_c)

    # Non-pose-only image IDs in sorted order — these have depth/ade entries.
    non_po_ids = [
        image_ids[i] for i in range(len(image_ids))
        if not (i < len(pose_only) and pose_only[i])
    ]

    for depth_pos, img_id in enumerate(non_po_ids):
        if idx is not None and depth_pos != idx:
            continue
        if depth_pos >= len(depths):
            break

        depth_field = depths[depth_pos]
        ade_field   = aes[depth_pos] if depth_pos < len(aes) else None

        depth_pil = _open_image_field(depth_field)
        ade_pil   = _open_image_field(ade_field)
        if depth_pil is None:
            continue

        if img_id not in colmap_cam_map:
            continue
        K_c, R_c, t_c = colmap_cam_map[img_id]

        ade_rgb = (np.array(ade_pil.convert("RGB"), dtype=np.uint8)
                   if ade_pil is not None else None)

        result = _unproject_depth(
            depth_pil, ade_rgb,
            K_np=K_c.astype(np.float64).copy(),
            R_np=R_c.astype(np.float64),
            t_np=t_c.astype(np.float64),
            target_size=target_size,
            depth_scale=depth_scale,
            max_depth=max_depth,
            skip_colors_arr=skip_colors_arr,
        )
        if result is None:
            continue
        pts_world, r_ch, g_ch, b_ch = result

        colors = [f"rgb({r},{g},{b})" for r, g, b in zip(r_ch, g_ch, b_ch)]
        fig.add_trace(go.Scatter3d(
            x=pts_world[:, 0],
            y=pts_world[:, 1],
            z=pts_world[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=colors, line_width=0),
            name=f"colmap_depth_{img_id}",
            showlegend=False,
        ))
