import os
import pickle
import numpy as np
import reducer.support.navigator as nav
from reducer.config import modelpath
from reducer.support.exceptions import TargetNotFoundError, AlgorithmError, InputError

########################################################
#                   Useful Functions                   #
########################################################

def combine_dict(d1, d2):
    return {
        k: tuple(d[k] for d in (d1, d2) if k in d)
        for k in set(d1.keys()) | set(d2.keys())
    }

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

def upsample(x, n): return np.repeat(x, n, axis=0)

def downsample(x, n):
    return np.array([x[n*i: n*(i+1)].mean() for i in range(len(x)//n - 1)])

def train_val_split(datadic, p):
    traindic, valdic = {}, {}
    for key in datadic.keys():
        N = len(datadic[key])
        N_train = int(N*p)

        traindic[key] = datadic[key][:N_train]
        valdic[key] = datadic[key][N_train:]
    return traindic, valdic

def dump(dic, foldername, filename, motherpath=modelpath):
    if not os.path.exists(os.path.join(motherpath, foldername)): os.mkdir(os.path.join(motherpath, foldername))
    with open(os.path.join(motherpath, foldername, filename), "wb") as f: pickle.dump(dic, f)

def to_string(l): return [str(i) for i in l]

########################################################
#                   Loading Functions                  #
########################################################

def param_finder(string, target, sep="_", sep2="="):
    params = string.split(sep)
    for pm in params:
        if target in pm:
            if sep2 != None: return pm.split(sep2)[1]
            else: return pm
    raise TargetNotFoundError(f"{target} not found in string {string}.")

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
    elif type(specify) == int:
        print(f"Loading model {specify}, i.e. seed={list(rnns.keys())[specify]}")
        return list(rnns.values())[specify]
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

def fit_loader(specify, episode):
    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f:
        data = pickle.load(f)
    return data

def simulation_loader(specify, tpe, episode="random"):
    seed = seed_order[specify]
    print(f"Loading model {specify}, i.e. seed={seed}")
    if episode == "random": episode = str(np.random.choice(240))
    else: episode = str(episode)
    print(f"Using episode {episode}")
    files = nav.file_finder(target=[seed, tpe, episode], extension=".npy", parent_name=os.path.join(modelpath, "observations"))
    tpe_full = param_finder(files[0], "tpe")
    fname = f"seed={seed}_tpe={tpe_full}_episode={episode}.npy"

    dic = {}
    for item in ["observations", "actions", "activities_rnn", "activities_MLP1"]:
        filepath = os.path.join(modelpath, item, fname)
        dic[item] = np.load(filepath)

    return dic

########################################################
#                      Parameters                      #
########################################################

seed_order = ['2760377', '3199993', '9781ba', '541058', '3307e9']

########################################################
#                       Fitting                        #
########################################################

class FitFuncs():
    def __init__(self):
        self.dic = {
            "ssine": single_sine,
            "esine": envelope_sine,
            "constant": constant
        }

        self.rdic = {} # reverse dict
        for key, value in self.dic.items(): self.rdic[value] = key

    def __call__(self, name, reverse=False):
        if not reverse: return self.dic[name]
        else: return self.rdic[name]

def single_sine(t, A, f, phi, b, s):
    return A*np.sin(f*t + phi) + b + s*t

def envelope_sine(t, A, f_slow, phi_slow, f_fast, phi_fast, b, s):
    return A*np.sin(f_slow*t + phi_slow)*np.sin(f_fast*t + phi_fast) + b + s*t

def constant(t, b): return np.ones(len(t))*b

class FitGenerator():
    def __init__(self, dic):
        self.dic = dic
        self.translator = FitFuncs()

    def eliminate_drift(self, funcname, params): # implemented for ssine only
        if funcname == "ssine":
            params[-1] = 0
            return params
        return params

    def generate(self, t, eliminate_drift=False):
        seqs = []
        for key in ["C", "y", "x"]:
            pms, funcname, _ = self.dic[key]
            if eliminate_drift: pms = self.eliminate_drift(funcname, pms)
            seq = self.translator(funcname)(t, *pms)
            seqs.append(seq)
        
        return np.vstack(seqs).T






# Development purposes
if __name__ == "__main__":
    dic = simulation_loader(0, "constant")
    import pdb; pdb.set_trace()