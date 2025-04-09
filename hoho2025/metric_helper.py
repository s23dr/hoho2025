import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import trimesh
from time import time

MAX_SCORE = 1.0

def get_one_primitive(p1, p2, c=(255, 0, 0), radius=25, primitive_type='cylinder', sections=6):
    if len(c) == 1:
        c = [c[0]] * 4
    elif len(c) == 3:
        c = [*c, 255]
    elif len(c) != 4:
        raise ValueError(f'{c} is not a valid color (must have 1,3, or 4 elements).')

    p1, p2 = np.asarray(p1), np.asarray(p2)
    l = np.linalg.norm(p2 - p1)
    
    # Add check for zero-length edges
    if l < 1e-6:
        return None
        
    direction = (p2 - p1) / l

    T = np.eye(4)
    T[:3, 2] = direction
    T[:3, 3] = (p1 + p2) / 2

    b0, b1 = T[:3, 0], T[:3, 1]
    if np.abs(np.dot(b0, direction)) < np.abs(np.dot(b1, direction)):
        T[:3, 1] = -np.cross(b0, direction)
    else:
        T[:3, 0] = np.cross(b1, direction)

    if primitive_type == 'capsule':
        mesh = trimesh.primitives.Capsule(radius=radius, height=l, transform=T, sections=sections)
    elif primitive_type == 'cylinder':
        mesh = trimesh.primitives.Cylinder(radius=radius, height=l, transform=T, sections=sections)
    else:
        raise ValueError("Unknown primitive!")

    # Add vertex color initialization check
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4)) * 255
        
    mesh.visual.vertex_colors = np.ones_like(mesh.visual.vertex_colors) * c
    return mesh

def get_primitives(vertices, edges, radius=25, c=[255, 0, 0]):
    # Convert vertices to a NumPy array
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    else:
        vertices = np.asarray(vertices)
    
    # Convert edges to a NumPy array of integers
    if isinstance(edges, torch.Tensor):
        edges = edges.detach().cpu().numpy().astype(np.int64)
    else:
        edges = np.asarray(edges, dtype=np.int64)
    
    primitives = []
    for e in edges:
        # Add edge validation
        if e[0] >= len(vertices) or e[1] >= len(vertices):
            continue
        primitive = get_one_primitive(vertices[e[0]], vertices[e[1]], radius=radius, c=c)
        if primitive is not None:
            primitives.append(primitive)
    return primitives



def compute_mesh_iou_VOLUME(pd_vertices, pd_edges, gt_vertices, gt_edges, radius=20, engine='manifold'):
    # check empty 
    if len(pd_edges) == 0 or len(gt_edges) == 0:
        return 0.0

    pd_vertices = pd_vertices.detach().cpu() if isinstance(pd_vertices, torch.Tensor) else pd_vertices
    pd_edges = pd_edges.detach().cpu() if isinstance(pd_edges, torch.Tensor) else pd_edges
    gt_vertices = gt_vertices.detach().cpu() if isinstance(gt_vertices, torch.Tensor) else gt_vertices
    gt_edges = gt_edges.detach().cpu() if isinstance(gt_edges, torch.Tensor) else gt_edges

    pd_primitives = get_primitives(pd_vertices, pd_edges, radius=radius, c=[0, 255, 0])
    gt_primitives = get_primitives(gt_vertices, gt_edges, radius=radius, c=[255, 0, 0])
    # check for empty primitives
    if not pd_primitives or not gt_primitives:
        return 0.0

    # Add bounding box check to detect non-overlapping cases quickly
    pd_bounds = np.array([p.bounds for p in pd_primitives])
    gt_bounds = np.array([p.bounds for p in gt_primitives])
    
    pd_min, pd_max = np.min(pd_bounds[:, 0], axis=0), np.max(pd_bounds[:, 1], axis=0)
    gt_min, gt_max = np.min(gt_bounds[:, 0], axis=0), np.max(gt_bounds[:, 1], axis=0)
    
    # If bounding boxes don't overlap, return 0
    if np.any(pd_max < gt_min) or np.any(pd_min > gt_max):
        return 0.0
    t=time()
    mesh_pred = trimesh.boolean.union(pd_primitives, engine=engine)
    #print(f"mesh_pred union: {time() - t} {mesh_pred.is_volume}")
    t=time()
    mesh_gt= trimesh.boolean.union(gt_primitives, engine=engine)
    #print(f"mesh_gt union: {time() - t} {mesh_gt.is_volume}")

    if mesh_pred.is_volume and mesh_gt.is_volume:
        t=time()
        inter_volume = trimesh.boolean.intersection([mesh_pred, mesh_gt], engine=engine).volume
        #print(f"inter_volume: {time() - t}")
    else:
        all_inter = []
        t=time()
        for pd_prim in pd_primitives:
            pd_min, pd_max = pd_prim.bounds
            for gt_prim in gt_primitives:
                # Skip intersection calculation if bounding boxes don't overlap
                gt_min, gt_max = gt_prim.bounds
                if np.any(pd_max < gt_min) or np.any(pd_min > gt_max):
                    continue
                inter = trimesh.boolean.intersection([pd_prim, gt_prim], engine=engine)
                if inter.is_volume and inter.volume > 0:
                    all_inter.append(inter)
        inter_volume = trimesh.boolean.union(all_inter, engine=engine).volume if all_inter else 0
        #print(f"all_inter: {time() - t}")
    union_volume = mesh_pred.volume + mesh_gt.volume - inter_volume
    
    return inter_volume / union_volume if union_volume > 0 else 0.0


# ----------------- Corner F1 -----------------
def compute_ap_metrics(pd_vertices, gt_vertices, thresh=25):
    if len(pd_vertices) == 0 or len(gt_vertices) == 0:
        return 0.0

    dists = cdist(pd_vertices, gt_vertices)
    row_ind, col_ind = linear_sum_assignment(dists)

    tp = (dists[row_ind, col_ind] <= thresh).sum()
    precision = tp / len(pd_vertices) if len(pd_vertices) > 0 else 0
    recall = tp / len(gt_vertices) if len(gt_vertices) > 0 else 0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return f1

def batch_corner_f1(X, Y, distance_thresh=25):
    results = []
    for (pd_v, _), (gt_v, _) in zip(X, Y):
        results.append(compute_ap_metrics(pd_v, gt_v, thresh=distance_thresh))
    return np.array(results)

# ----------------- HSS Metric -----------------
from collections import namedtuple
HSSReturnType = namedtuple('HSSReturnType', ['hss', 'f1', 'iou'])
def hss(y_hat_v, y_hat_e, y_v, y_e, vert_thresh=0.5, edge_thresh=0.5):
    X = [(y_hat_v, y_hat_e)]
    Y = [(y_v, y_e)]
    t=time()
    f1 = np.clip(batch_corner_f1(X, Y, distance_thresh=vert_thresh)[0], 0, 1)
    #print(f"f1 {f1}: in {time() - t:.2f} sec")
    t=time()
    IoU = np.clip(compute_mesh_iou_VOLUME(y_hat_v, y_hat_e, y_v, y_e, radius=edge_thresh), 0, 1)
    #print(f"IoU: {IoU} in {time() - t:.2f} sec")
    score = 2 * f1 * IoU / (f1 + IoU) if (f1 + IoU) > 0 else 0.0
    return HSSReturnType(hss=score, f1=f1, iou=IoU)