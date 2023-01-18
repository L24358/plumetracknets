import os
from reducer.config import modelpath

def file_pointer(target, parent_name):
    for dirpath, dirnames, filenames in os.walk(os.path.join(modelpath, parent_name)):
        if target in filenames: return os.path.join(dirpath, target)
    
    raise FileNotFoundError(f"{target} not found in {parent_name} under modelpath (set in config).")

def file_finder(target="VRNN", extension=".pt", parent_name=modelpath):
    res = []
    for filename in os.listdir(parent_name):
        cond1 = (filename[-len(extension):] == extension) if extension != None else True
        cond2 = (target in filename) if target != None else True
        if cond1 and cond2: res.append(os.path.join(parent_name, filename))
    return res

def purge(folder, target): pass