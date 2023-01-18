from reducer.support.exceptions import TargetNotFoundError

def param_finder(string, target, sep="_", sep2="="):
    params = string.split(sep)
    for pm in params:
        if target in pm:
            if sep2 != None: return pm.split(sep2)[1]
            else: return pm
    raise TargetNotFoundError(f"{target} not found in string {string}.")

def regroup_by_(dic, idx=0): # TODO: More elegant solution uses pandas
    res = {}
    for key, value in dic.items():
        sortby = key[idx]
        if sortby not in res.keys(): res[sortby] = {}
        key = list(key)
        key.pop(idx)
        res[sortby][tuple(key)] = value
    return res