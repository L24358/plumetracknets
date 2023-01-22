import os
import numpy as np
import reducer.support.navigator as nav
from reducer.config import modelpath
from reducer.support.exceptions import TargetNotFoundError, AlgorithmError, InputError

def param_finder(string, target, sep="_", sep2="="):
    params = string.split(sep)
    for pm in params:
        if target in pm:
            if sep2 != None: return pm.split(sep2)[1]
            else: return pm
    raise TargetNotFoundError(f"{target} not found in string {string}.")

def dict_to_generator(dic):
    for value in dic.values():
        yield value

def different(x1, x2, threshold=1e-7):
    diff = np.array(x1) - np.array(x2)
    norm = pow(diff, 2).sum()
    if norm > threshold: return True, norm
    else: return False, norm

def get_bins(data, nbins):
    minn, maxx = min(data), max(data)
    return np.linspace(minn, maxx, nbins+1)

def model_loader(specify="all"):
    rnns = {}
    for mtype in ["weight_hh", "weight_ih", "bias_hh", "bias_ih"]:
        modelnames = nav.file_finder(
            target=mtype,
            extension=".npy",
            parent_name=os.path.join(modelpath, "rnn_matrix"))
        for modelname in modelnames:
            key = param_finder(modelname, "seed", sep2=None)[4:]
            if key not in rnns.keys(): rnns[key] = []
            rnns[key].append(np.load(modelname))

    for key in rnns.keys():
        if len(rnns[key]) != 4: raise AlgorithmError("Did not find complete set of weight/bias matrices!")

    # Return value depends on specify
    if specify == "all": return rnns
    elif specify == "random":
        idx = np.random.choice(len(rnns.keys()))
        print(f"Loading model {idx+1}.")
        return list(rnns.values())[idx]
    elif type(specify) == int: return list(rnns.values())[specify]
    else: raise InputError(f"keyword `specify` does not support {specify}.")

def actor_loader(specify="all"):
    nns = {}
    for mtype in ["base.actor1.0.weight", "base.actor1.0.bias", "base.actor.0.weight", "base.actor.0.bias",\
        "dist.fc_mean.weight", "dist.fc_mean.bias", "dist.logstd._bias"]:
        modelnames = nav.file_finder(
            target=mtype,
            extension=".npy",
            parent_name=os.path.join(modelpath, "actors"))
        for modelname in modelnames:
            key = param_finder(modelname, "seed", sep2=None)[4:]
            if key not in nns.keys(): nns[key] = {}
            nns[key][mtype] = np.load(modelname)

    for key in nns.keys():
        if len(nns[key]) != 7: raise AlgorithmError("Did not find complete set of weight/bias matrices!")

    # Return value depends on specify
    if specify == "all": return nns
    elif specify == "random":
        idx = np.random.choice(len(nns.keys()))
        print(f"Loading model {idx+1}.")
        return list(nns.values())[idx]
    elif type(specify) == int: return list(nns.values())[specify]
    else: raise InputError(f"keyword `specify` does not support {specify}.")