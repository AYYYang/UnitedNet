from os.path import dirname, basename, isfile, join
import glob
import os

modules = glob.glob(join(dirname(__file__), "*.py"))
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        # Use os.path.basename to properly get the filename regardless of OS
        cur_f = os.path.basename(f)[:-3]
        exec(f"from src.configs.{cur_f} import *")