import os
from reducer.config import modelpath
from reducer.support.exceptions import TargetNotFoundError

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

def param_finder(string, target, sep="_", sep2="="):
    params = string.split(sep)
    for pm in params:
        if target in pm:
            if sep2 != None: return pm.split(sep2)[1]
            else: return pm
    raise TargetNotFoundError(f"{target} not found in string {string}.")

def purge(folder, target): pass