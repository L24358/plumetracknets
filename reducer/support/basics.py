import os
import pickle
import numpy as np
import reducer.support.navigator as nav
from itertools import product
from sklearn.metrics.cluster import mutual_info_score as MIscore
from sklearn.metrics.cluster import normalized_mutual_info_score as aMIscore
from reducer.config import modelpath, graphpath
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

def reorder_dict(dic):
    return {k: dic[k] for k in seed_order}

def fjoin(*args, tpe=1):
    motherpath = modelpath if tpe==1 else graphpath
    folders = args[:-1]
    os.makedirs(os.path.join(motherpath, *folders), exist_ok=True)
    return os.path.join(motherpath, *args)

def npsave(item, *args): np.save(os.path.join(modelpath, *args), item)

def npload(*args): return np.load(os.path.join(modelpath, *args))

def pklsave(item, *args):
    f = open(os.path.join(modelpath, *args), "wb")
    pickle.dump(item, f)
    f.close()

def pklload(*args):
    f = open(os.path.join(modelpath, *args), "rb")
    data = pickle.load(f)
    f.close()
    return data

def sort_by_val(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

def update_by_add(dic1, dic2):
    dic3 = {}
    for key in dic1.keys(): dic3[key] = dic1[key] + dic2[key]
    return dic3

########################################################
#                Very Specific Functions               #
########################################################

def concat_dic_values(dic):
    res = []
    for key in sorted(dic.keys()):
        res += list(dic[key])
    return res

def strip_rp(l):
    new = []
    prev = None
    for t in range(len(l)):
        if l[t] != prev:
            new.append(l[t])
            prev = l[t]
    return new

def get_grams(l, elements, n=2): # only works if len(elements) is small, else too expensive
    dic = {}
    grams = product(elements, repeat=n)
    for gram in grams: dic[gram] = 0

    for i in range(len(l)-n+1):
        pair = tuple(l[i:i+n])
        dic[pair] += 1
    return dic

def get_regime(dic_regime, epi, t, vals=["tracking", "recovery", "lost"]):
    dic = {"tracking": vals[0], "recovery": vals[1], "lost": vals[2]}
    epi, t = int(epi), int(t)
    for reg in ["tracking", "recovery", "lost"]:
        if dic_regime[epi][reg][t]: return dic[reg]
    raise bcs.AlgorithmError()

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
    rnns = reorder_dict(rnns)
    if specify == "all": return rnns
    elif specify == "random":
        idx = np.random.choice(len(rnns.keys()))
        print(f"Loading model {idx+1}.")
        return list(rnns.values())[idx]
    elif type(specify) == int:
        print(f"Loading model {specify}, i.e. seed={list(rnns.keys())[specify]}")
        return list(rnns.values())[specify]
    else: raise InputError(f"keyword `specify` does not support {specify}.")

def actor_loader(specify="all", verbose=True):
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
    nns = reorder_dict(nns)
    if specify == "all": return nns
    elif specify == "random":
        idx = np.random.choice(len(nns.keys()))
        if verbose: print(f"Loading model {idx}, i.e. seed={list(nns.keys())[idx]}")
        return list(nns.values())[idx]
    elif type(specify) == int:
        if verbose: print(f"Loading model {specify}, i.e. seed={list(nns.keys())[specify]}")
        return list(nns.values())[specify]
    else: raise InputError(f"keyword `specify` does not support {specify}.")

def fit_loader(specify, episode): # TODO: need to rerun after reorder_dict implementation
    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f:
        data = pickle.load(f)
    return data

def simulation_loader(specify, tpe, episode="random", verbose=True):
    seed = seed_order[specify]
    if verbose: print(f"Loading model {specify}, i.e. seed={seed}")
    if episode == "random": episode = str(np.random.choice(240))
    else: episode = str(episode)
    if verbose: print(f"Using episode {episode}")
    files = nav.file_finder(target=[seed, tpe, episode], extension=".npy", parent_name=os.path.join(modelpath, "observations"))
    tpe_full = param_finder(files[0], "tpe")
    fname = f"seed={seed}_tpe={tpe_full}_episode={episode}.npy"

    dic = {}
    for item in ["observations", "actions", "activities_rnn", "activities_MLP1"]:
        filepath = os.path.join(modelpath, item, fname)
        dic[item] = np.load(filepath)

    return dic

########################################################
#                     Read / Write                     #
########################################################

def dump(dic, foldername, filename, motherpath=modelpath):
    if not os.path.exists(os.path.join(motherpath, foldername)): os.mkdir(os.path.join(motherpath, foldername))
    with open(os.path.join(motherpath, foldername, filename), "wb") as f: pickle.dump(dic, f)

def to_string(l): return [str(i) for i in l]

def to_(l, types): return [types[i](l[i]) for i in range(len(l))]

def split_sort(row, sep=" ", sep2="="):
    l = row.strip().split(sep)
    if sep2 != None: return [item.split(sep2)[-1] for item in l]
    return l

def write_row(filename, foldername, row):
    os.makedirs(os.path.join(modelpath, foldername), exist_ok=True)
    f = open(os.path.join(modelpath, foldername, filename), "a")
    f.write(row + "\n")
    f.close()

def read_row(filename, foldername, action):
    f = open(os.path.join(modelpath, foldername, filename), "r")
    data = f.readlines()
    f.close()

    res = []
    for row in data: res.append(action(row))
    return res

########################################################
#                      Parameters                      #
########################################################

seed_order = ['2760377', '3199993', '9781ba', '541058', '3307e9']

track_dic_manual = {38:30, 16:0, 21:20, 91:50, 6:0, 7:150, 8:150, 9:100, 11:50, 12:100, 14:50, 15:30, 17:30, 19:70}

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


########################################################
#                 Mutual Information                   #
########################################################

def time_shift_MI_wrap(stimulus, response, shift, ds, numstate, normalization=False):
    stimulus = easy_resample(stimulus, numstate)
    response = easy_resample(response, numstate)
    fMI, aMI = time_shift_MI(stimulus, response, shift, ds)
    if not normalization: return fMI
    else: return aMI

def time_shift_MI(stimulus, response, shift, ds):
    assert len(stimulus) == len(response)
    past_MI, future_MI, past_aMI, future_aMI = [], [], [], []
    for sh in range(ds, shift, ds): 
        past_MI.append(MIscore(stimulus[:-sh], response[sh:])) #negative shift
        past_aMI.append(aMIscore(stimulus[:-sh], response[sh:]))
        future_MI.append(MIscore(stimulus[sh:], response[:-sh])) #positive shift
        future_aMI.append(aMIscore(stimulus[sh:], response[:-sh]))
    now_MI, now_aMI = MIscore(stimulus, response), aMIscore(stimulus, response)
    all_MI = list(reversed(past_MI))+[now_MI]+future_MI
    all_aMI = list(reversed(past_aMI))+[now_aMI]+future_aMI
    return all_MI, all_aMI

def easy_resample(stimulus, numstate):
	sort_sti = list(sorted(stimulus))
	chunk = int(len(stimulus)/numstate)
	mmdic = {}
	for n in range(numstate-1):
		mmdic[(sort_sti[chunk*n], sort_sti[chunk*(n+1)])] = n
	new_sti = []
	for s in stimulus:
		flag = True
		for key in mmdic.keys():
			if s >= key[0] and s < key[1]:
				new_sti.append(mmdic[key])
				flag = False
		if flag: new_sti.append(numstate-1)
	return new_sti

########################################################
#                 Coordinate Transforms                #
########################################################

class Agent():
    def __init__(self):
        self.loc = np.zeros(2) # origin
        self.angle = np.pi # absolute angle; facing towards -y
        self.history = np.array([[*self.loc, self.angle]])

    def update(self, r, theta):
        self.loc += r*np.array([np.cos(theta), np.sin(theta)])
        self.angle += theta
        self.angle = self.angle % np.pi
        return self.loc, self.angle

    def log(self, overwrite=[]):
        if overwrite == []: overwrite = [*self.loc, self.angle]
        self.history = np.append(self.history, np.array([overwrite]), axis=0)

    def set_angle(self, theta): self.angle = theta

def get_wind(observations, actions):
    agent = Agent()
    ego_wind_angles = []
    abs_wind_angles = []
    for t in range(len(observations)):
        x, y, C = observations[t]
        r, theta = actions[t]

        ego_wind_angle = np.arctan2(y, x)
        abs_wind_angle = agent.angle + ego_wind_angle
        ego_wind_angles.append(ego_wind_angle % np.pi)
        abs_wind_angles.append(abs_wind_angle % np.pi)

        agent.update(r, theta)
        agent.log()

    return ego_wind_angles, abs_wind_angles, agent.history

class PCASVD():
    def __init__(self, r):
        self.r = r
        self.U = np.load(os.path.join(modelpath, "pca_frame", "U.npy"))
        self.V = np.load(os.path.join(modelpath, "pca_frame", "V.npy"))
        self.S, self.singular_values_ = load_S()
        self.col_mean = np.load(os.path.join(modelpath, "pca_frame", "col_mean.npy"))
        self.components_ = self.V.T # the rows are the components, as per sklearn convention

    def transform(self, X): # It is transformed into the mean-subtracted coordinates
        X -= self.col_mean
        return X[:, :self.r] @ self.V[:self.r, :self.r]

    def get_base_traj(self, r=None):
        if r == None: r = self.r
        # return (self.U @ self.S @ self.V.T)[:, :r]
        return self.U[:, :r] @ self.S[:r, :r] @ self.V.T[:r, :r]

def load_S():
    s = np.load(os.path.join(modelpath, "pca_frame", "s.npy"))
    S = np.zeros((30184, 64))
    np.fill_diagonal(S, s)
    return S, s



# Development purposes
if __name__ == "__main__":
    dic = simulation_loader(0, "constant")
    import pdb; pdb.set_trace()