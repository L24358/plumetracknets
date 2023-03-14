import os
import numpy as np
from reducer.config import modelpath

def all_in(targets, filename):
    return np.all([target in filename for target in targets])

def file_pointer(target, parent_name):
    for dirpath, dirnames, filenames in os.walk(os.path.join(modelpath, parent_name)):
        if target in filenames: return os.path.join(dirpath, target)
    
    raise FileNotFoundError(f"{target} not found in {parent_name} under modelpath (set in config).")

def file_finder(target="VRNN", extension=".pt", parent_name=modelpath):
    res = []
    for filename in os.listdir(parent_name):
        # correct extension
        cond1 = (filename[-len(extension):] == extension) if extension != None else True

        # correct targets
        if target == None: cond2 = True
        elif type(target) == list: cond2 = all_in(target, filename)
        else: cond2 = all_in([target], filename)

        # select those with both conditions fulfilled
        if cond1 and cond2: res.append(os.path.join(parent_name, filename))
    return res

def purge(folder, target): pass