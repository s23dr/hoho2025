from .hoho import *
from . import vis

import importlib
import sys

class LazyLoadModule:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattribute__(self, attr):
        if attr == 'module_name' or attr == 'module':
            return super().__getattribute__(attr)

        if self.module is None:
            self.module = importlib.import_module(f'hoho.{self.module_name}')
            sys.modules[self.module_name] = self.module

        return getattr(self.module, attr)
    
try:
    import viz3d
except ImportError:
    viz3d = LazyLoadModule('viz3d')
