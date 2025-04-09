import os 
import json
import shutil
from pathlib import Path
from typing import Dict
import warnings 
import contextlib
import tempfile 
from PIL import Image
import io
import webdataset as wds 
import numpy as np
import importlib
import subprocess


from PIL import ImageFile

from huggingface_hub.utils._headers import build_hf_headers # note: using _headers


ImageFile.LOAD_TRUNCATED_IMAGES = True

LOCAL_DATADIR = None

def setup(local_dir='./data/usm-training-data/data'):
    
    # If we are in the test environment, we need to link the data directory to the correct location
    tmp_datadir = Path('/tmp/data/data')
    local_test_datadir = Path('./data/usm-test-data-x/data')
    local_val_datadir = Path(local_dir)
    
    os.system('pwd')
    os.system('ls -lahtr .')
    
    if tmp_datadir.exists() and not local_test_datadir.exists():
        global LOCAL_DATADIR
        LOCAL_DATADIR = local_test_datadir
        # shutil.move(datadir, './usm-test-data-x/data')
        print(f"Linking {tmp_datadir} to {LOCAL_DATADIR} (we are in the test environment)")
        LOCAL_DATADIR.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_DATADIR.symlink_to(tmp_datadir)
    else:
        LOCAL_DATADIR = local_val_datadir
        print(f"Using {LOCAL_DATADIR} as the data directory (we are running locally)")
    
    if not LOCAL_DATADIR.exists():
        warnings.warn(f"Data directory {LOCAL_DATADIR} does not exist: creating it...")
        LOCAL_DATADIR.mkdir(parents=True)
    
    return LOCAL_DATADIR


def download_package(package_name, path_to_save='packages'):
    """
    Downloads a package using pip and saves it to a specified directory.

    Parameters:
    package_name (str): The name of the package to download.
    path_to_save (str): The path to the directory where the package will be saved.
    """
    try:
        # pip download webdataset -d packages/webdataset --platform manylinux1_x86_64 --python-version 38 --only-binary=:all:
        subprocess.check_call([subprocess.sys.executable, "-m", "pip", "download", package_name, 
                               "-d", str(Path(path_to_save)/package_name),  # Download the package to the specified directory
                               "--platform", "manylinux1_x86_64",  # Specify the platform
                               "--python-version", "38",  # Specify the Python version
                               "--only-binary=:all:"])  # Download only binary packages
        print(f'Package "{package_name}" downloaded successfully')
    except subprocess.CalledProcessError as e:
        print(f'Failed to downloaded package "{package_name}". Error: {e}')
              
              
def install_package_from_local_file(package_name, folder='packages'):
    """
    Installs a package from a local .whl file or a directory containing .whl files using pip.

    Parameters:
    path_to_file_or_directory (str): The path to the .whl file or the directory containing .whl files.
    """
    try:
        pth = str(Path(folder) / package_name)
        subprocess.check_call([subprocess.sys.executable, "-m", "pip", "install", 
                               "--no-index",  # Do not use package index
                               "--find-links", pth,  # Look for packages in the specified directory or at the file
                               package_name])  # Specify the package to install
        print(f"Package installed successfully from {pth}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package from {pth}. Error: {e}")
        
        
def importt(module_name, as_name=None):
    """
    Imports a module and returns it.

    Parameters:
    module_name (str): The name of the module to import.
    as_name (str): The name to use for the imported module. If None, the original module name will be used.

    Returns:
    The imported module.
    """
    for _ in range(2):
        try:
            if as_name is None:
                print(f'imported {module_name}')
                return importlib.import_module(module_name)
            else:
                print(f'imported {module_name} as {as_name}')
                return importlib.import_module(module_name, as_name)
        except ModuleNotFoundError as e:
            install_package_from_local_file(module_name)
            print(f"Failed to import module {module_name}. Error: {e}")
            
    
def prepare_submission():
    # Download packages from requirements.txt 
    if Path('requirements.txt').exists():
        print('downloading packages from requirements.txt')
        Path('packages').mkdir(exist_ok=True)
        with open('requirements.txt') as f:
            packages = f.readlines()
            for p in packages:
                download_package(p.strip())
                
    print('all packages downloaded. Don\'t foget to include the packages in the submission by adding them with git lfs.')
        

def Rt_to_eye_target(im, K, R, t):
    height = im.height
    focal_length = K[0,0]
    fov = 2.0 * np.arctan2((0.5 * height), focal_length) / (np.pi / 180.0)

    x_axis, y_axis, z_axis = R

    eye = -(R.T @ t).squeeze()
    z_axis = z_axis.squeeze()
    target = eye + z_axis
    up = -y_axis
    
    return eye, target, up, fov        


########## general utilities ##########


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        
@contextlib.contextmanager
def temp_working_directory():
    with tempfile.TemporaryDirectory(dir='.') as D:
        with working_directory(D):
            yield


############# Dataset ############# 
def proc(row, split='train'):
    out = {}
    out['__key__'] = None
    out['__imagekey__'] = []
    for k, v in row.items():
        key_parts = k.split('.')
        colname = key_parts[0]
        if colname == 'ade20k': 
            out['__imagekey__'].append(key_parts[1])
        if colname in {'ade20k', 'depthcm', 'gestalt'}:
            if colname in out:
                out[colname].append(v)
            else:
                out[colname] = [v]
        elif colname in {'wireframe', 'mesh'}:
            out.update({a: b for a,b in v.items()})
        elif colname in 'kr':
            out[colname.upper()] = v
        else:
            out[colname] = v
    return Sample(out)



def decode_colmap(s):
    import hoho2025.read_write_colmap as read_write_colmap
    with temp_working_directory():

        with open('points3D.bin', 'wb') as stream:
            stream.write(s['points3d'])


        with open('cameras.bin', 'wb') as stream:
            stream.write(s['cameras'])


        with open('images.bin', 'wb') as stream:
            stream.write(s['images'])
    
        
        cameras, images, points3D = read_write_colmap.read_model(
            path='.', ext='.bin'
        )
    return cameras, images, points3D


def decode(row):
    cameras, images, points3D = decode_colmap(row)
    
    out = {}
    
    for k, v in row.items():
        # colname = k.split('.')[0]
        if k in {'ade20k', 'depthcm', 'gestalt'}:
            # print(k, len(v), type(v))
            v = [Image.open(io.BytesIO(im)) for im in v]
            if k in out:
                out[k].extend(v)
            else:
                out[k] = v
        elif k in {'wireframe', 'mesh'}:
            # out.update({a: b.tolist() for a,b in v.items()})
            v = dict(np.load(io.BytesIO(v)))
            out.update({a: b for a,b in v.items()})
        elif k in 'kr':
            out[k.upper()] = v
        elif k == 'cameras':
            out[k] = cameras
        elif k == 'images':
            out[k] = images
        elif k =='points3d':
            out[k] = points3D
        else:
            out[k] = v
            
    return Sample(out)


class Sample(Dict):
    def __repr__(self):
        return str({k: v.shape if hasattr(v, 'shape') else [type(v[0])] if isinstance(v, list) else type(v) for k,v in self.items()})

    
        
def get_params():
    exmaple_param_dict = {
        "competition_id": "usm3d/S23DR",
        "competition_type": "script",
        "metric": "custom",
        "token": "hf_**********************************",
        "team_id": "local-test-team_id",
        "submission_id": "local-test-submission_id",
        "submission_id_col": "__key__",
        "submission_cols": [
            "__key__",
            "wf_edges",
            "wf_vertices",
            "edge_semantics"
        ],
        "submission_rows": 180,
        "output_path": ".",
        "submission_repo": "<THE HF MODEL ID of THIS REPO",
        "time_limit": 7200,
        "dataset": "usm3d/usm-test-data-x",
        "submission_filenames": [
            "submission.parquet"
        ]
    }
    
    param_path = Path('params.json')
    
    if not param_path.exists():
        print('params.json not found (this means we probably aren\'t in the test env). Using example params.')
        params = exmaple_param_dict
    else:
        print('found params.json (this means we are probably in the test env). Using params from file.')
        with param_path.open() as f:
            params = json.load(f)
    print(params)
    return params





SHARD_IDS = {'train': (0, 25), 'val': (25, 26), 'public': (26, 27), 'private': (27, 32)}
def get_dataset(decode='pil', proc=proc, split='train', dataset_type='webdataset', stream=True):
    if LOCAL_DATADIR is None:
        raise ValueError('LOCAL_DATADIR is not set. Please run setup() first.')
        
    local_dir = Path(LOCAL_DATADIR)
    if split != 'all':
        local_dir = local_dir / split
    
    paths = [str(p) for p in local_dir.rglob('*.tar.gz')]
    msg = f'no tarfiles found in {local_dir}.'
    if len(paths) == 0:
        if stream:
            if split=='all': split = 'train'
            warnings.warn('streaming isn\'t using with \'all\': changing `split` to \'train\'')
            warnings.warn(msg)
            if split == 'val':
                names = [f'data/val/inputs/hoho_v3_{i:03}-of-032.tar.gz' for i in range(*SHARD_IDS[split])]
            elif split == 'train':
                names = [f'data/train/hoho_v3_{i:03}-of-032.tar.gz' for i in range(*SHARD_IDS[split])]
            
            auth = build_hf_headers()['authorization']
            paths = [f"pipe:curl -L -s https://huggingface.co/datasets/usm3d/hoho-train-set/resolve/main/{name} -H 'Authorization: {auth}'" for name in names]
        else:
            raise FileNotFoundError(msg)
    
    dataset = wds.WebDataset(paths)
        
    if decode is not None:
        dataset = dataset.decode(decode)
    else:
        dataset = dataset.decode()
    
    dataset = dataset.map(proc)
    
    if dataset_type == 'webdataset':
        return dataset
    
    if dataset_type == 'hf':
        import datasets
        from datasets import Features, Value, Sequence, Image, Array2D
       
        if split == 'train':
            return datasets.IterableDataset.from_generator(lambda: dataset.iterator())
        elif split == 'val':
            return datasets.IterableDataset.from_generator(lambda: dataset.iterator())
        else:
            raise NotImplementedError('only train and val are implemented as hf datasets')

    
    