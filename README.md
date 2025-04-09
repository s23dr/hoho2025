---
license: apache-2.0
---
# HoHo2025 Tools

Tools and utilities for the [S23DR-2025 competition](https://huggingface.co/spaces/usm3d/S23DR2025) and [HoHo25k Dataset](https://huggingface.co/datasets/usm3d/hoho25k)

## Installation 

### pip install over http
```bash
pip install git+http://hf.co/usm3d/tools2025.git    
```

or editable 
```bash
git clone http://hf.co/usm3d/tools2025 
cd tools2025
pip install -e .
```

### Usage example

```python
from datasets import load_dataset
from hoho2025.vis import plot_all_modalities
from hoho2025.viz3d import *

def read_colmap_rec(colmap_data):
    import pycolmap
    import tempfile,zipfile
    import io
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(colmap_data), "r") as zf:
            zf.extractall(tmpdir)  # unpacks cameras.txt, images.txt, etc. to tmpdir
        # Now parse with pycolmap
        rec = pycolmap.Reconstruction(tmpdir)
        return rec

ds = load_dataset("usm3d/hoho25k", streaming=True, trust_remote_code=True)
for a in ds['train']:
    break

fig, ax = plot_all_modalities(a)

## Now 3d

fig3d = init_figure()
plot_reconstruction(fig3d, read_colmap_rec(a['colmap_binary']))
plot_wireframe(fig3d, a['wf_vertices'], a['wf_edges'], a['wf_classifications'])
plot_bpo_cameras_from_entry(fig3d, a)
fig3d
```

## Example wireframe estimation 

Look in [hoho2025/example_solution.py](hoho2025/example_solution.py)

```python
from hoho2025.example_solutions import predict_wireframe
pred_vertices, pred_connections = predict_wireframe(a)

fig3d = init_figure()
plot_reconstruction(fig3d, read_colmap_rec(a['colmap_binary']))
plot_wireframe(fig3d, pred_vertices, pred_connections, color='rgb(0, 0, 255)')
fig3d
```


And to get the metric

```python
from hoho2025.metric_helper import hss

score = hss(pred_vertices, pred_connections, a['wf_vertices'], a['wf_edges'], vert_thresh=0.5, edge_thresh=0.5)
print (score)
```