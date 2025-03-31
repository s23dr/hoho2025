from setuptools import setup, find_packages
import glob


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='hoho',
	version='0.0.4',
	description='Tools and utilites for the HoHo Dataset and S23DR Competition',
	url='usm3d.github.io',
	author='Jack Langerman, Dmytro Mishkin, S23DR Orgainizing Team',
	author_email='hoho@jackml.com',
	install_requires=required,
	packages=find_packages(),
	include_package_data=True)