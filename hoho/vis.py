import trimesh
import numpy as np
from copy import deepcopy
from PIL import Image

from . import color_mappings

def line(p1, p2, c=(255,0,0), resolution=10, radius=0.05):
    '''draws a 3d cylinder along the line (p1, p2)'''
    # check colors
    if len(c) == 1:
        c = [c[0]]*4
    elif len(c) == 3:
        c = [*c, 255]
    elif len(c) != 4:
        raise ValueError(f'{c} is not a valid color (must have 1,3, or 4 elements).')
        
    # compute length and direction of segment
    p1, p2 = np.asarray(p1), np.asarray(p2)
    l = np.linalg.norm(p2-p1)
    
    direction = (p2 - p1) / l
    
    # point z along direction of segment
    T = np.eye(4)
    T[:3, 2] = direction
    T[:3, 3] = (p1+p2)/2
    
    #reorthogonalize basis
    b0, b1 = T[:3, 0], T[:3, 1]
    if np.abs(np.dot(b0, direction)) < np.abs(np.dot(b1, direction)):
        T[:3, 1] = -np.cross(b0, direction)
    else:
        T[:3, 0] = np.cross(b1, direction)
    
    # generate and transform mesh
    mesh = trimesh.primitives.Cylinder(radius=radius, height=l, transform=T)
    
    # apply uniform color
    mesh.visual.vertex_colors = np.ones_like(mesh.visual.vertex_colors)*c
         
    return mesh

def show_wf(row, radius=10, show_vertices=False, vertex_color=(255,0,0, 255)):
    EDGE_CLASSES = ['eave',
                    'ridge',
                    'step_flashing',
                    'rake',
                    'flashing',
                    'post',
                    'valley',
                    'hip',
                    'transition_line']
    out_meshes = []
    if show_vertices:
        out_meshes.extend([trimesh.primitives.Sphere(radius=radius+5, center = center, color=vertex_color) for center in row['wf_vertices']])
        for m in out_meshes:
            m.visual.vertex_colors = np.ones_like(m.visual.vertex_colors)*vertex_color
    if 'edge_semantics' not in row:
        print ("Warning: edge semantics is not here, skipping")
        out_meshes.extend([line(a,b, radius=radius, c=(214, 251, 248)) for a,b in np.stack([*row['wf_vertices']])[np.stack(row['wf_edges'])]])
    elif len(np.stack(row['wf_edges'])) ==  len(row['edge_semantics']):
        out_meshes.extend([line(a,b, radius=radius, c=color_mappings.gestalt_color_mapping[EDGE_CLASSES[cls_id]]) for (a,b), cls_id in zip(np.stack([*row['wf_vertices']])[np.stack(row['wf_edges'])], row['edge_semantics'])])
    else:
        print ("Warning: edge semantics has different length compared to edges, skipping semantics")
        out_meshes.extend([line(a,b, radius=radius, c=(214, 251, 248)) for a,b in np.stack([*row['wf_vertices']])[np.stack(row['wf_edges'])]])
    return out_meshes
    # return [line(a,b, radius=radius, c=color_mappings.edge_colors[cls_id]) for (a,b), cls_id in zip(np.stack([*row['wf_vertices']])[np.stack(row['wf_edges'])], row['edge_semantics'])]


def show_grid(edges, meshes=None, row_length=5):
    '''
        edges: list of list of meshes
        meshes: optional corresponding list of meshes
        row_length: number of meshes per row
  
        returns trimesh.Scene()
    '''
    
    T = np.eye(4)
    out = []
    edges = [sum(e[1:], e[0]) for e in edges]
    row_height = 1.1 * max((e.extents for e in edges), key=lambda e: e[1])[1]
    col_width = 1.1 * max((e.extents for e in edges), key=lambda e: e[0])[0]
    # print(row_height, col_width)
    
    if meshes is None:
        meshes = [None]*len(edges)

    for i, (gt, mesh) in enumerate(zip(edges, meshes), start=0):
        mesh = deepcopy(mesh)
        gt = deepcopy(gt)

        if i%row_length != 0:
            T[0, 3] += col_width

        else:
            T[0, 3] = 0
            T[1, 3] += row_height

        # print(T[0,3]/col_width, T[2,3]/row_height)
        
        if mesh is not None:
            mesh.apply_transform(T)
            out.append(mesh)
                            
        gt.apply_transform(T)
        out.append(gt)
        
                            
        out.extend([mesh, gt])

            
    return trimesh.Scene(out)




def visualize_order_images(row_order):
    return create_image_grid(row_order['ade20k'] + row_order['gestalt'] + [visualize_depth(dm) for dm in row_order['depthcm']], num_per_row=len(row_order['ade20k']))

def create_image_grid(images, target_length=312, num_per_row=2):
    # Calculate the target size for the first image
    first_img = images[0]
    aspect_ratio = first_img.width / first_img.height
    new_width = int((target_length ** 2 * aspect_ratio) ** 0.5)
    new_height = int((target_length ** 2 / aspect_ratio) ** 0.5)
    
    # Resize the first image
    resized_images = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) for img in images]
    
    # Calculate the grid size
    num_rows = (len(resized_images) + num_per_row - 1) // num_per_row
    grid_width = new_width * num_per_row
    grid_height = new_height * num_rows
    
    # Create a new image for the grid
    grid_img = Image.new('RGB', (grid_width, grid_height))
    
    # Paste the images into the grid
    for i, img in enumerate(resized_images):
        x_offset = (i % num_per_row) * new_width
        y_offset = (i // num_per_row) * new_height
        grid_img.paste(img, (x_offset, y_offset))
    
    return grid_img


import matplotlib.pyplot as plt

def visualize_depth(depth, min_depth=None, max_depth=None, cmap='rainbow'):
    depth = np.array(depth)
    
    if min_depth is None:
        min_depth = np.min(depth)
    if max_depth is None:
        max_depth = np.max(depth)
    
    
    # Normalize the depth to be between 0 and 1
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.clip(depth, 0, 1)
    
    # Use the matplotlib colormap to convert the depth to an RGB image
    cmap = plt.get_cmap(cmap)
    depth_image = (cmap(depth) * 255).astype(np.uint8)
    
    # Convert the depth image to a PIL image
    depth_image = Image.fromarray(depth_image)
    
    return depth_image