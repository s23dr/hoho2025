from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np 


def preregister_mean_std(verts_to_transform, target_verts, single_scale=True):
    mu_target = target_verts.mean(axis=0)
    mu_in = verts_to_transform.mean(axis=0)
    std_target = np.std(target_verts, axis=0)
    std_in = np.std(verts_to_transform, axis=0)
    
    if np.any(std_in == 0):
        std_in[std_in == 0] = 1
    if np.any(std_target == 0):
        std_target[std_target == 0] = 1
    if np.any(np.isnan(std_in)):
        std_in[np.isnan(std_in)] = 1
    if np.any(np.isnan(std_target)):
        std_target[np.isnan(std_target)] = 1
        
    if single_scale:
        std_target = np.linalg.norm(std_target)
        std_in = np.linalg.norm(std_in)
    
    transformed_verts = (verts_to_transform - mu_in) / std_in
    transformed_verts = transformed_verts * std_target + mu_target
    
    return transformed_verts


def update_cv(cv, gt_vertices):
    if cv < 0:
        diameter = cdist(gt_vertices, gt_vertices).max()
        # Cost of adding or deleting a vertex is set to -cv times the diameter of the ground truth wireframe
        cv = -cv * diameter
    return cv

def compute_WED(pd_vertices, pd_edges, gt_vertices, gt_edges, cv_ins=-1/2, cv_del=-1/4, ce=1.0, normalized=True, preregister=True, single_scale=True):
    '''The function computes the Wireframe Edge Distance (WED) between two graphs.
    pd_vertices: list of predicted vertices
    pd_edges: list of predicted edges
    gt_vertices: list of ground truth vertices
    gt_edges: list of ground truth edges
    cv_ins: vertex insertion cost: if positive, the cost in centimeters of inserting vertex, if negative, multiplies diameter to compute cost (default is -1/2)
    cv_del: vertex deletion cost: if positive, the cost in centimeters of deleting a vertex, if negative, multiplies diameter to compute cost (default is -1/4)
    ce: edge cost (multiplier of the edge length for edge deletion and insertion, default is 1.0)
    normalized: if True, the WED is normalized by the total length of the ground truth edges
    preregister: if True, the predicted vertices have their mean and scale matched to the ground truth vertices
    '''
    
    pd_vertices = np.array(pd_vertices)
    gt_vertices = np.array(gt_vertices)
    pd_edges = np.array(pd_edges)
    gt_edges = np.array(gt_edges)        
    
        
    cv_del = update_cv(cv_del, gt_vertices)
    cv_ins = update_cv(cv_ins, gt_vertices)
    
    # Step 0: Prenormalize / preregister
    if preregister:
        pd_vertices = preregister_mean_std(pd_vertices, gt_vertices, single_scale=single_scale)
    

    # Step 1: Bipartite Matching
    distances = cdist(pd_vertices, gt_vertices, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(distances)
        
    
    # Step 2: Vertex Translation
    translation_costs = np.sum(distances[row_ind, col_ind])
    
    # Step 3: Vertex Deletion
    unmatched_pd_indices = set(range(len(pd_vertices))) - set(row_ind)
    deletion_costs = cv_del * len(unmatched_pd_indices)  
    
    # Step 4: Vertex Insertion
    unmatched_gt_indices = set(range(len(gt_vertices))) - set(col_ind)
    insertion_costs = cv_ins * len(unmatched_gt_indices)  
    
    # Step 5: Edge Deletion and Insertion
    updated_pd_edges = [(col_ind[np.where(row_ind == edge[0])[0][0]], col_ind[np.where(row_ind == edge[1])[0][0]]) for edge in pd_edges if len(edge)==2 and edge[0] in row_ind and edge[1] in row_ind]
    pd_edges_set = set(map(tuple, [set(edge) for edge in updated_pd_edges]))
    gt_edges_set = set(map(tuple, [set(edge) for edge in gt_edges]))

    
    # Delete edges not in ground truth
    edges_to_delete = pd_edges_set - gt_edges_set
    
    vert_tf = [np.where(col_ind == v)[0][0] if v in col_ind else 0 for v in range(len(gt_vertices))]
    deletion_edge_costs = ce * sum(np.linalg.norm(pd_vertices[vert_tf[edge[0]]] - pd_vertices[vert_tf[edge[1]]]) for edge in edges_to_delete if len(edge) == 2)

    
    # Insert missing edges from ground truth
    edges_to_insert = gt_edges_set - pd_edges_set
    insertion_edge_costs = ce * sum(np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]]) for edge in edges_to_insert if len(edge) == 2) 
    
    # Step 6: Calculation of WED
    WED = translation_costs + deletion_costs + insertion_costs + deletion_edge_costs + insertion_edge_costs

    
    if normalized:
        total_length_of_gt_edges = np.linalg.norm((gt_vertices[gt_edges[:, 0]] - gt_vertices[gt_edges[:, 1]]), axis=1).sum()
        WED = WED / total_length_of_gt_edges
        
    # print ("Total length", total_length_of_gt_edges)
    return WED