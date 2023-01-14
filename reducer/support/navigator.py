import os
from reducer.config import modelpath

def file_pointer(target, parent_name):
    for dirpath, dirnames, filenames in os.walk(os.path.join(modelpath, parent_name)):
        if target in filenames: return os.path.join(dirpath, target)
    
    raise FileNotFoundError(f"{target} not found in {parent_name} under modelpath (set in config).")