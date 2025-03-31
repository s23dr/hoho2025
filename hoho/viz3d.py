
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
"""
# Slightly modified by Dmytro Mishkin

from typing import Optional
import numpy as np
import pycolmap
import plotly.graph_objects as go


### Some helper functions for geometry
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def to_homogeneous(points):
    pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)

def t_to_proj_center(qvec, tvec):
    Rr = qvec2rotmat(qvec)
    tt = (-Rr.T) @ tvec
    return tt

def calib(params):
    out = np.eye(3)
    if len(params) == 3:
        out[0,0] = params[0]
        out[1,1] = params[0]
        out[0,2] = params[1]
        out[1,2] = params[2]
    else:
        out[0,0] = params[0]
        out[1,1] = params[1]
        out[0,2] = params[2]
        out[1,2] = params[3]
    return out


### Plotting functions

def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
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
    traces = [go.Scatter3d(x=x1, y=y1, z=z1,
                        mode='lines',
                        line=dict(color=color, width=2)) for x1, y1, z1 in zip(x,y,z)]
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
    """Plot a camera frustum from pose and intrinsic matrix."""
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
    intr = calib(camera.params)
    if intr[0][0] > 10000:
        print("Bad camera")
        return
    plot_camera(
        fig,
        qvec2rotmat(image.qvec).T,
        t_to_proj_center(image.qvec, image.tvec),
        intr,#calibration_matrix(),
        name=name or str(image.id),
        **kwargs)


def plot_cameras(
        fig: go.Figure,
        reconstruction,#: pycolmap.Reconstruction,
        **kwargs):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction["images"].items():
        plot_camera_colmap(
            fig, image, reconstruction["cameras"][image.camera_id], **kwargs)


def plot_reconstruction(
        fig: go.Figure,
        rec,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        points: bool = True,
        cameras: bool = True,
        cs: float = 1.0,
        single_color_points=False,
        camera_color='rgba(0, 255, 0, 0.5)'):
    # rec is result of loading reconstruction from "read_write_colmap.py"
    # Filter outliers
    xyzs = []
    rgbs = []
    for k, p3D in rec['points'].items():
        xyzs.append(p3D.xyz)
        rgbs.append(p3D.rgb)

    if points:
        plot_points(fig, np.array(xyzs), color=color if single_color_points else np.array(rgbs), ps=1, name=name)
    if cameras:
        plot_cameras(fig, rec, color=camera_color, legendgroup=name, size=cs)


def plot_pointcloud(
        fig: go.Figure,
        pts: np.ndarray,
        colors: np.ndarray,
        ps: int = 2,
        name: Optional[str] = None):
    """Plot a set of 3D points."""
    plot_points(fig, np.array(pts), color=colors, ps=ps, name=name)


def plot_triangle_mesh(
        fig: go.Figure,
        vert: np.ndarray,
        colors: np.ndarray,
        triangles: np.ndarray,
        name: Optional[str] = None):
    """Plot a triangle mesh."""
    tr = go.Mesh3d(
        x=vert[:,0],
        y=vert[:,1],
        z=vert[:,2],
        vertexcolor = np.clip(255*colors, 0, 255),
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=triangles[:,0],
        j=triangles[:,1],
        k=triangles[:,2],
        name=name,
        showscale=False
    )
    fig.add_trace(tr)

def plot_estimate_and_gt(pred_vertices, pred_connections, gt_vertices=None, gt_connections=None):
    fig3d = init_figure()
    c1 = (30, 20, 255)
    img_color = [c1 for _ in range(len(pred_vertices))]
    plot_points(fig3d, pred_vertices, color = img_color, ps = 10)  
    lines = []
    for c in pred_connections:
        v1 = pred_vertices[c[0]]
        v2 = pred_vertices[c[1]]
        lines.append(np.stack([v1, v2], axis=0))
    plot_lines_3d(fig3d, np.array(lines), img_color, ps=4)  
    if gt_vertices is not None:
        c2 = (30, 255, 20)
        img_color2 = [c2 for _ in range(len(gt_vertices))]
        plot_points(fig3d, gt_vertices, color = img_color2, ps = 10)  
        if gt_connections is not None:
            gt_lines = []
            for c in gt_connections:
                v1 = gt_vertices[c[0]]
                v2 = gt_vertices[c[1]]
                gt_lines.append(np.stack([v1, v2], axis=0))
        plot_lines_3d(fig3d, np.array(gt_lines), img_color2, ps=4)        
    fig3d.show()
    return fig3d
