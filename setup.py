from setuptools import setup, find_packages
import glob
import os

# Try to read from requirements.txt, but have fallback
try:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'requirements.txt')) as f:
        required = f.read().splitlines()
except FileNotFoundError:
    # Fallback to hardcoded dependencies
    required = [
        'datasets',
        'huggingface-hub',
        'ipywidgets',
        'matplotlib',
        'numpy',
        'opencv-python',
        'Pillow',
        'plotly',
        'pycolmap',
        'scipy',
        'torch',
        'trimesh',
        'webdataset==0.2.111',
    ]

setup(name='hoho2025',
	version='0.1.0',
	description='Tools and utilites for the HoHo Dataset and S23DR Competition',
	url='https://github.com/s23dr/hoho2025',
	author='Jack Langerman, Dmytro Mishkin, S23DR Orgainizing Team',
	author_email='hoho@jackml.com',
	install_requires=required,
	packages=find_packages(),
	python_requires='>=3.10',
	include_package_data=True)