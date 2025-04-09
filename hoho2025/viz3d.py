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

def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.FigureWidget()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            eye=dict(x=0., y=-.1, z=-2),
            up=dict(x=0, y=-1., z=0),
            projection=dict(type="orthographic")),
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
        #print (p3D)
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
    """Plot a camera as a cone with camera frustum."""
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


def plot_bpo_cameras_from_entry(fig: go.Figure, entry: dict, idx = None):
    def cam2world_to_world2cam(R, t):
        rt = np.eye(4)
        rt[:3,:3] = R
        rt[:3,3] = t.reshape(-1)
        rt = np.linalg.inv(rt)
        return rt[:3,:3], rt[:3,3]
    
    for i in range(len(entry['R'])):
        if idx is not None and i != idx:
            continue
        K = np.array(entry['K'][i])
        R = np.array(entry['R'][i])
        t = np.array(entry['t'][i])
        R, t = cam2world_to_world2cam(R, t)
        plot_camera(fig, R, t, K)
    
    
