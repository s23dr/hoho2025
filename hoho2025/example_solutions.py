# Description: This file contains the handcrafted solution for the task of wireframe reconstruction 
import io
import tempfile
import zipfile
from collections import defaultdict
from typing import Tuple, List
import cv2
import numpy as np
import pycolmap
from PIL import Image as PImage
from scipy.spatial.distance import cdist

from hoho2025.color_mappings import ade20k_color_mapping, gestalt_color_mapping


def empty_solution():
    '''Return a minimal valid solution, i.e. 2 vertices and 1 edge.'''
    return np.zeros((2,3)), [(0, 1)]
    
    
def read_colmap_rec(colmap_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(colmap_data), "r") as zf:
            zf.extractall(tmpdir)  # unpacks cameras.txt, images.txt, etc. to tmpdir
        # Now parse with pycolmap
        rec = pycolmap.Reconstruction(tmpdir)
        return rec

def convert_entry_to_human_readable(entry):
    out = {}
    for k, v in entry.items():
        if 'colmap' in k:
            out[k] = read_colmap_rec(v)
        elif k in ['wf_vertices', 'wf_edges', 'K', 'R', 't', 'depth']:
            out[k] = np.array(v)
        else:
            out[k]=v
    out['__key__'] = entry['order_id']
    return out


def get_house_mask(ade20k_seg):
    """
    Get a mask of the house in the ADE20K segmentation map.
    """
    house_classes_ade20k = [
        'wall',
        'house',
        'building;edifice',
        'door;double;door',
        'windowpane;window',
    ]
    np_seg = np.array(ade20k_seg)
    full_mask = np.zeros(np_seg.shape[:2], dtype=np.uint8)
    for c in house_classes_ade20k:
        color = np.array(ade20k_color_mapping[c])
        mask = cv2.inRange(np_seg, color-0.5, color+0.5)
        full_mask = np.logical_or(full_mask, mask)
    return full_mask


def point_to_segment_dist(pt, seg_p1, seg_p2):
    """
    Computes the Euclidean distance from pt to the line segment p1->p2.
    pt, seg_p1, seg_p2: (x, y) as np.ndarray
    """
    # If both endpoints are the same, just return distance to one of them
    if np.allclose(seg_p1, seg_p2):
        return np.linalg.norm(pt - seg_p1)
    seg_vec = seg_p2 - seg_p1
    pt_vec = pt - seg_p1
    seg_len2 = seg_vec.dot(seg_vec)
    t = max(0, min(1, pt_vec.dot(seg_vec)/seg_len2))
    proj = seg_p1 + t*seg_vec
    return np.linalg.norm(pt - proj)


def get_vertices_and_edges_from_segmentation(gest_seg_np, edge_th=25.0):
    """
    Identify apex and eave-end vertices, then detect lines for eave/ridge/rake/valley.
    For each connected component, we do a line fit with cv2.fitLine, then measure
    segment endpoints more robustly. We then associate apex points that are within
    'edge_th' of the line segment. We record those apex–apex connections for edges
    if at least 2 apexes lie near the same component line.
    """
    #--------------------------------------------------------------------------------
    # Step A: Collect apex and eave_end vertices
    #--------------------------------------------------------------------------------
    if not isinstance(gest_seg_np, np.ndarray):
        gest_seg_np = np.array(gest_seg_np)
    vertices = []
    # Apex
    apex_color = np.array(gestalt_color_mapping['apex'])
    apex_mask = cv2.inRange(gest_seg_np, apex_color-0.5, apex_color+0.5)
    if apex_mask.sum() > 0:
        output = cv2.connectedComponentsWithStats(apex_mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        stats, centroids = stats[1:], centroids[1:]  # skip background
        for i in range(numLabels-1):
            vert = {"xy": centroids[i], "type": "apex"}
            vertices.append(vert)

    # Eave end
    eave_end_color = np.array(gestalt_color_mapping['eave_end_point'])
    eave_end_mask = cv2.inRange(gest_seg_np, eave_end_color-0.5, eave_end_color+0.5)
    if eave_end_mask.sum() > 0:
        output = cv2.connectedComponentsWithStats(eave_end_mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        stats, centroids = stats[1:], centroids[1:]
        for i in range(numLabels-1):
            vert = {"xy": centroids[i], "type": "eave_end_point"}
            vertices.append(vert)

    # Consolidate apex points as array:
    apex_pts = []
    apex_idx_map = []  # keep track of index in 'vertices'
    for idx, v in enumerate(vertices):
        apex_pts.append(v['xy'])
        apex_idx_map.append(idx)
    apex_pts = np.array(apex_pts)

    connections = []
    edge_classes = ['eave', 'ridge', 'rake', 'valley']
    for edge_class in edge_classes:
        edge_color = np.array(gestalt_color_mapping[edge_class])
        mask_raw = cv2.inRange(gest_seg_np, edge_color-0.5, edge_color+0.5)
        # Possibly do morphological open/close to avoid merges or small holes
        kernel = np.ones((5, 5), np.uint8)  # smaller kernel to reduce over-merge
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
        if mask.sum() == 0:
            continue

        # Connected components
        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        # skip the background
        stats, centroids = stats[1:], centroids[1:]
        label_indices = range(1, numLabels)

        # For each connected component, do a line fit
        for lbl in label_indices:
            ys, xs = np.where(labels == lbl)
            if len(xs) < 2:
                continue
            # Fit a line using cv2.fitLine
            pts_for_fit = np.column_stack([xs, ys]).astype(np.float32)
            # (vx, vy, x0, y0) = direction + a point on the line
            line_params = cv2.fitLine(pts_for_fit, distType=cv2.DIST_L2, 
                                      param=0, reps=0.01, aeps=0.01)
            vx, vy, x0, y0 = line_params.ravel()
            # We'll approximate endpoints by projecting (xs, ys) onto the line,
            # then taking min and max in the 1D param along the line.

            # param along the line = ( (x - x0)*vx + (y - y0)*vy )
            proj = ( (xs - x0)*vx + (ys - y0)*vy )
            proj_min, proj_max = proj.min(), proj.max()
            p1 = np.array([x0 + proj_min*vx, y0 + proj_min*vy])
            p2 = np.array([x0 + proj_max*vx, y0 + proj_max*vy])

            #--------------------------------------------------------------------------------
            # Step C: If apex points are within 'edge_th' of segment, they are connected
            #--------------------------------------------------------------------------------
            if len(apex_pts) < 2:
                continue

            # Distance from each apex to the line segment
            dists = np.array([
                point_to_segment_dist(apex_pts[i], p1, p2)
                for i in range(len(apex_pts))
            ])

            # Indices of apex points that are near
            near_mask = (dists <= edge_th)
            near_indices = np.where(near_mask)[0]
            if len(near_indices) < 2:
                continue

            # Connect each pair among these near apex points
            for i in range(len(near_indices)):
                for j in range(i+1, len(near_indices)):
                    a_idx = near_indices[i]
                    b_idx = near_indices[j]
                    # 'a_idx' and 'b_idx' are indices in apex_pts / apex_idx_map
                    vA = apex_idx_map[a_idx]
                    vB = apex_idx_map[b_idx]
                    # Store the connection using sorted indexing
                    conn = tuple(sorted((vA, vB)))
                    connections.append(conn)

    return vertices, connections


def get_uv_depth(vertices: List[dict],
                 depth_fitted: np.ndarray,
                 sparse_depth: np.ndarray,
                 search_radius: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each vertex, returns a 2D array of (u,v) and a matching 1D array of depths.
    
    We attempt to use the sparse_depth if available in a local neighborhood:
      1. For each vertex coordinate (x, y), define a local window in sparse_depth 
         of size (2*search_radius + 1).
      2. Collect all valid (nonzero) values in that window.
      3. If any exist, we take the *closest* valid pixel's depth.
      4. Otherwise, we use depth_fitted[y, x].
    
    Parameters
    ----------
    vertices : List[dict]
        Each dict must have "xy" at least, e.g. {"xy": (x, y), ...}
    depth_fitted : np.ndarray
        A 2D array (H, W), the dense (or corrected) depth for fallback.
    sparse_depth : np.ndarray
        A 2D array (H, W), mostly zeros except where accurate data is available.
    search_radius : int
        Pixel radius around the vertex in which to look for sparse depth values.
    
    Returns
    -------
    uv : np.ndarray of shape (N, 2)
        2D float coordinates of each vertex (x, y).
    vertex_depth : np.ndarray of shape (N,)
        Depth value chosen for each vertex.
    """
    
    # Collect each vertex's (x, y)
    uv = np.array([vert['xy'] for vert in vertices], dtype=np.float32)
    
    # Convert to integer pixel coordinates (round or floor)
    uv_int = np.round(uv).astype(np.int32)
    H, W = depth_fitted.shape[:2]
    
    # Clip coordinates to stay within image bounds
    uv_int[:, 0] = np.clip(uv_int[:, 0], 0, W - 1)
    uv_int[:, 1] = np.clip(uv_int[:, 1], 0, H - 1)
    
    # Prepare output array of depths
    vertex_depth = np.zeros(len(vertices), dtype=np.float32)
    dense_count = 0
    
    for i, (x_i, y_i) in enumerate(uv_int):
        # Local region in [x_i - search_radius, x_i + search_radius]
        x0 = max(0, x_i - search_radius)
        x1 = min(W, x_i + search_radius + 1)
        y0 = max(0, y_i - search_radius)
        y1 = min(H, y_i + search_radius + 1)
        
        # Crop out the local window in sparse_depth
        region = sparse_depth[y0:y1, x0:x1]
        
        # Find all valid (non-zero) depths
        valid_mask = (region > 0)
        valid_y, valid_x = np.where(valid_mask)
        
        if valid_y.size > 0:
            # Compute global coordinates for each valid pixel
            global_x = x0 + valid_x
            global_y = y0 + valid_y
            
            # Compute squared distance to center (x_i, y_i)
            dist_sq = (global_x - x_i)**2 + (global_y - y_i)**2
            
            # Find the nearest valid pixel
            min_idx = np.argmin(dist_sq)
            nearest_depth = region[valid_y[min_idx], valid_x[min_idx]]
            vertex_depth[i] = nearest_depth
        else:
            # Fallback to the dense depth
            vertex_depth[i] = depth_fitted[y_i, x_i]
            dense_count += 1
    return uv, vertex_depth



def project_vertices_to_3d(uv: np.ndarray, depth_vert: np.ndarray, col_img: pycolmap.Image) -> np.ndarray:
    """
    Projects 2D vertex coordinates with associated depths to 3D world coordinates.

    Parameters
    ----------
    uv : np.ndarray
        (N, 2) array of 2D vertex coordinates (u, v).
    depth_vert : np.ndarray
        (N,) array of depth values for each vertex.
    col_img : pycolmap.Image

    Returns
    -------
    vertices_3d : np.ndarray
        (N, 3) array of vertex coordinates in 3D world space.
    """
    # Backproject to 3D local camera coordinates
    xy_local = np.ones((len(uv), 3))
    K = col_img.camera.calibration_matrix()
    xy_local[:, 0] = (uv[:, 0] - K[0, 2]) / K[0, 0]
    xy_local[:, 1] = (uv[:, 1] - K[1, 2]) / K[1, 1]
    # Get the 3D vertices
    vertices_3d_local = xy_local * depth_vert[...,None]
    
    # Create camera-to-world transformation matrix
    world_to_cam = np.eye(4)
    world_to_cam[:3] = col_img.cam_from_world.matrix()
    cam_to_world = np.linalg.inv(world_to_cam)
    
    # Transform local 3D points to world coordinates
    vertices_3d_homogeneous = cv2.convertPointsToHomogeneous(vertices_3d_local)
    vertices_3d = cv2.transform(vertices_3d_homogeneous, cam_to_world)
    vertices_3d = cv2.convertPointsFromHomogeneous(vertices_3d).reshape(-1, 3)
    return vertices_3d


def create_3d_wireframe_single_image(vertices: List[dict],
                                     connections: List[Tuple[int, int]],
                                     depth: PImage,
                                     colmap_rec: pycolmap.Reconstruction,
                                     img_id: str,
                                     ade_seg: PImage) -> np.ndarray:
    """
    Processes a single image view to generate 3D vertex coordinates from existing 2D vertices/edges.

    Parameters
    ----------
    vertices : List[dict]
        List of 2D vertex dictionaries (e.g., {"xy": (x, y), "type": ...}).
    connections : List[Tuple[int, int]]
        List of 2D edge connections (indices into the vertices list).
    depth : PIL.Image
        Initial dense depth map as a PIL Image.
    colmap_rec : pycolmap.Reconstruction
        COLMAP reconstruction data.
    img_id : str
        Identifier for the current image within the COLMAP reconstruction.
    ade_seg : PIL.Image
        ADE20k segmentation map for the image.

    Returns
    -------
    vertices_3d : np.ndarray
        (N, 3) array of vertex coordinates in 3D world space.
        Returns an empty array if processing fails (e.g., missing sparse depth).
    """
    # Check if initial vertices/connections are valid
    if (len(vertices) < 2) or (len(connections) < 1):
        # This case should ideally be handled before calling, but good to double check.
        print(f'Warning: create_3d_wireframe_single_image called with insufficient vertices/connections for image {img_id}')
        return np.empty((0, 3))

    # Get fitted dense depth and sparse depth
    depth_fitted, depth_sparse, found_sparse, col_img = get_fitted_dense_depth(
        depth, colmap_rec, img_id, ade_seg
    )

    # Get UV coordinates and depth for each vertex
    uv, depth_vert = get_uv_depth(vertices, depth_fitted, depth_sparse, 10)

    # Backproject to 3D
    vertices_3d = project_vertices_to_3d(uv, depth_vert, col_img)

    return vertices_3d


def merge_vertices_3d(vert_edge_per_image, th=0.5):
    '''Merge vertices that are close to each other in 3D space and are of same types'''
    # Initialize structures to collect vertices and connections from all images
    all_3d_vertices = []
    connections_3d = []
    all_indexes = []
    cur_start = 0
    types = []
    
    # Combine vertices and update connection indices across all images
    for cimg_idx, (vertices, connections, vertices_3d) in vert_edge_per_image.items():
        types += [int(v['type']=='apex') for v in vertices]
        all_3d_vertices.append(vertices_3d)
        connections_3d+=[(x+cur_start,y+cur_start) for (x,y) in connections]
        cur_start+=len(vertices_3d)
    all_3d_vertices = np.concatenate(all_3d_vertices, axis=0)
    
    # Calculate distance matrix between all vertices
    distmat = cdist(all_3d_vertices, all_3d_vertices)
    types = np.array(types).reshape(-1,1)
    same_types = cdist(types, types)
    
    # Create mask for vertices that should be merged (close in space and same type)
    mask_to_merge = (distmat <= th) & (same_types==0)
    new_vertices = []
    new_connections = []
    
    # Extract vertex indices to merge based on the mask
    to_merge = sorted(list(set([tuple(a.nonzero()[0].tolist()) for a in mask_to_merge])))
    
    # Build groups of vertices to merge (transitive grouping)
    to_merge_final = defaultdict(list)
    for i in range(len(all_3d_vertices)):
        for j in to_merge:
            if i in j:
                to_merge_final[i]+=j
    
    # Remove duplicates in each group
    for k, v in to_merge_final.items():
        to_merge_final[k] = list(set(v))
    
    # Create final merge groups without duplicates
    already_there = set() 
    merged = []
    for k, v in to_merge_final.items():
        if k in already_there:
            continue
        merged.append(v)
        for vv in v:
            already_there.add(vv)
    
    # Calculate new vertex positions (average of merged groups)
    old_idx_to_new = {}
    count=0
    for idxs in merged:
        new_vertices.append(all_3d_vertices[idxs].mean(axis=0))
        for idx in idxs:
            old_idx_to_new[idx] = count
        count +=1
    new_vertices=np.array(new_vertices)
    
    # Update connections to use new vertex indices
    for conn in connections_3d:
        new_con = sorted((old_idx_to_new[conn[0]], old_idx_to_new[conn[1]]))
        if new_con[0] == new_con[1]:
            continue
        if new_con not in new_connections:
            new_connections.append(new_con)
    return new_vertices, new_connections


def prune_not_connected(all_3d_vertices, connections_3d, keep_largest=True):
    """
    Prune vertices not connected to anything. If keep_largest=True, also
    keep only the largest connected component in the graph.
    """
    if len(all_3d_vertices) == 0:
        return np.array([]), []

    # adjacency
    adj = defaultdict(set)
    for (i, j) in connections_3d:
        adj[i].add(j)
        adj[j].add(i)

    # keep only vertices that appear in at least one edge
    used_idxs = set()
    for (i, j) in connections_3d:
        used_idxs.add(i)
        used_idxs.add(j)

    if not used_idxs:
        return np.empty((0,3)), []

    # If we only want to remove truly isolated points, but keep multiple subgraphs:
    if not keep_largest:
        new_map = {}
        used_list = sorted(list(used_idxs))
        for new_id, old_id in enumerate(used_list):
            new_map[old_id] = new_id
        new_vertices = np.array([all_3d_vertices[old_id] for old_id in used_list])
        new_conns = []
        for (i, j) in connections_3d:
            if i in used_idxs and j in used_idxs:
                new_conns.append((new_map[i], new_map[j]))
        return new_vertices, new_conns

    # Otherwise find the largest connected component:
    visited = set()
    def bfs(start):
        queue = [start]
        comp = []
        visited.add(start)
        while queue:
            cur = queue.pop()
            comp.append(cur)
            for neigh in adj[cur]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)
        return comp

    # Collect all subgraphs
    comps = []
    for idx in used_idxs:
        if idx not in visited:
            c = bfs(idx)
            comps.append(c)

    # pick largest
    comps.sort(key=lambda c: len(c), reverse=True)
    largest = comps[0] if len(comps)>0 else []

    # Remap
    new_map = {}
    for new_id, old_id in enumerate(largest):
        new_map[old_id] = new_id

    new_vertices = np.array([all_3d_vertices[old_id] for old_id in largest])
    new_conns = []
    for (i, j) in connections_3d:
        if i in largest and j in largest:
            new_conns.append((new_map[i], new_map[j]))

    # remove duplicates
    new_conns = list(set([tuple(sorted(c)) for c in new_conns]))
    return new_vertices, new_conns

def get_sparse_depth(colmap_rec, img_id_substring, depth):
    """
    Return a sparse depth map for the COLMAP image whose name contains
    `img_id_substring`. The output is an array of shape `depth_shape` (H,W),
    where only the projected 3D points get a depth > 0, else 0.
    """
    H, W = depth.shape

    # 1) Find the matching COLMAP image
    found_img = None
    for img_id_c, col_img in colmap_rec.images.items():
        if img_id_substring in col_img.name:
            found_img = col_img
            break
    if found_img is None:
        print(f"Image substring {img_id_substring} not found in COLMAP.")
        return np.zeros((H, W), dtype=np.float32), False, None
    
    # 2) Gather 3D points that this image sees
    points_xyz = []
    for pid, p3D in colmap_rec.points3D.items():
        if found_img.has_point3D(pid):
            points_xyz.append(p3D.xyz)  # world coords
    if not points_xyz:
        print(f"No 3D points associated with {found_img.name}.")
        return np.zeros((H, W), dtype=np.float32), False, found_img
    
    points_xyz = np.array(points_xyz)  # (N, 3)
    
    # 3) For each point, project via col_img.project_point()
    uv = []
    z_vals = []
    for xyz in points_xyz:
        proj = found_img.project_point(xyz)  # returns (u, v) in image coords or None
        if proj is not None:
            u_i, v_i = proj
            u_i = int(round(u_i))
            v_i = int(round(v_i))
            # Check in-bounds
            if 0 <= u_i < W and 0 <= v_i < H:
                uv.append((u_i, v_i))
                # We'll compute depth as Z in camera coords
                # from the world->cam transform col_img holds
                mat4x4 = np.eye(4)
                mat4x4[:3, :4] = found_img.cam_from_world.matrix()
                p_cam =  mat4x4@ np.array([xyz[0], xyz[1], xyz[2], 1.0])
                z_vals.append(p_cam[2] / p_cam[3]) 
    
    uv = np.array(uv, dtype=int)     # shape (M,2)
    z_vals = np.array(z_vals)        # shape (M,)
    
    depth_out = np.zeros((H, W), dtype=np.float32)
    depth_out[uv[:,1], uv[:,0]] = z_vals  # Note: uv = (u, v), so row = v, col = u
    
    return depth_out, True, found_img


def fit_scale_robust_median(depth, sparse_depth, validity_mask=None):
    """
    Fit a scale factor to the depth map using the median of the ratio of sparse to dense depth.
    """
    if validity_mask is None:
        mask = (sparse_depth != 0)
    else:
        mask = (sparse_depth != 0) & validity_mask
    mask = mask & (depth <50) & (sparse_depth <50)
    X = depth[mask]
    Y = sparse_depth[mask]
    alpha =np.median(Y/X)
    depth_fitted = alpha * depth
    return alpha, depth_fitted
    

def get_fitted_dense_depth(depth, colmap_rec, img_id, ade20k_seg):
    """
    Gets sparse depth from COLMAP, computes a house mask, fits dense depth to sparse 
    depth within the mask, and returns the fitted dense depth.

    Parameters
    ----------
    depth : np.ndarray
        Initial dense depth map (H, W).
    colmap_rec : pycolmap.Reconstruction
        COLMAP reconstruction data.
    img_id : str
        Identifier for the current image within the COLMAP reconstruction.
    K : np.ndarray
        Camera intrinsic matrix (3x3).
    R : np.ndarray
        Camera rotation matrix (3x3).
    t : np.ndarray
        Camera translation vector (3,).
    ade20k_seg : PIL.Image
        ADE20k segmentation map for the image.

    Returns
    -------
    depth_fitted : np.ndarray
        Dense depth map scaled and shifted to align with sparse depth within the house mask (H, W).
    depth_sparse : np.ndarray
        The sparse depth map obtained from COLMAP (H, W).
    found_sparse : bool
        True if sparse depth points were found for this image, False otherwise.
    """
    depth_np = np.array(depth) / 1000. # Convert mm to meters if needed
    depth_sparse, found_sparse, col_img = get_sparse_depth(colmap_rec, img_id, depth_np)
    
    if not found_sparse:
        print(f'No sparse depth found for image {img_id}')
        # Return original (meter-scaled) depth if no sparse data
        return depth_np, np.zeros_like(depth_np), False, None

    # Get house mask to focus fitting on relevant areas
    house_mask = get_house_mask(ade20k_seg)
    
    # Fit dense depth to sparse depth (scale only), using only points within the house mask
    k, depth_fitted = fit_scale_robust_median(depth_np, depth_sparse, validity_mask=house_mask)
    print(f"Fitted depth scale k={k:.4f} for image {img_id}")
    #depth_fitted = depth_np# * house_mask.astype(np.float32)
    depth_sparse = depth_sparse# * house_mask.astype(np.float32)
    return depth_fitted, depth_sparse, True, col_img


def prune_too_far(all_3d_vertices, connections_3d, colmap_rec, th = 3.0):
    """
    Prune vertices that are too far from sparse point cloud
    
    """
    xyz_sfm=[]
    for k, v in colmap_rec.points3D.items():
        xyz_sfm.append(v.xyz)
    xyz_sfm = np.array(xyz_sfm)
    distmat = cdist(all_3d_vertices, xyz_sfm)
    mindist = distmat.min(axis=1)
    mask = mindist <= th
    all_3d_vertices_new = all_3d_vertices[mask]
    old_idx_survived = np.arange(len(all_3d_vertices))[mask]
    new_idxs = np.arange(len(all_3d_vertices_new))
    old_to_new_idx = dict(zip(old_idx_survived, new_idxs))
    connections_3d_new = [(old_to_new_idx[conn[0]], old_to_new_idx[conn[1]]) for conn in connections_3d if mask[conn[0]] and mask[conn[1]]]   
    return all_3d_vertices_new, connections_3d_new


def predict_wireframe(entry) -> Tuple[np.ndarray, List[int]]:
    """
    Predict 3D wireframe from a dataset entry.
    """
    good_entry = convert_entry_to_human_readable(entry)
    vert_edge_per_image = {}
    for i, (gest, depth, K, R, t, img_id, ade_seg) in enumerate(zip(good_entry['gestalt'],
                                                good_entry['depth'], 
                                                good_entry['K'],
                                                good_entry['R'],
                                                good_entry['t'],
                                                good_entry['image_ids'],
                                                good_entry['ade'] # Added ade20k segmentation
                                                )):
        colmap_rec = good_entry['colmap_binary']
        K = np.array(K)
        R = np.array(R)
        t = np.array(t)
        # Resize gestalt segmentation to match depth map size
        depth_size = (np.array(depth).shape[1], np.array(depth).shape[0]) # W, H
        gest_seg = gest.resize(depth_size)
        gest_seg_np = np.array(gest_seg).astype(np.uint8)
        
        # Get 2D vertices and edges first
        vertices, connections = get_vertices_and_edges_from_segmentation(gest_seg_np, edge_th=10.)
        
        # Check if we have enough to proceed
        if (len(vertices) < 2) or (len(connections) < 1):
            print(f'Not enough vertices or connections found in image {i}, skipping.')
            vert_edge_per_image[i] = [], [], np.empty((0, 3))
            continue
            
        # Call the refactored function to get 3D points
        vertices_3d = create_3d_wireframe_single_image(
            vertices, connections, depth, colmap_rec, img_id, ade_seg
        )
        # Store original 2D vertices, connections, and computed 3D points
        vert_edge_per_image[i] = vertices, connections, vertices_3d
    
    # Merge vertices from all images
    all_3d_vertices, connections_3d = merge_vertices_3d(vert_edge_per_image, 0.5)
    all_3d_vertices_clean, connections_3d_clean  = prune_not_connected(all_3d_vertices, connections_3d, keep_largest=False)
    all_3d_vertices_clean, connections_3d_clean  = prune_too_far(all_3d_vertices_clean, connections_3d_clean, colmap_rec, th = 4.0)
    
    if (len(all_3d_vertices_clean) < 2) or len(connections_3d_clean) < 1:
        print (f'Not enough vertices or connections in the 3D vertices')
        return empty_solution()

    return all_3d_vertices_clean, connections_3d_clean 
