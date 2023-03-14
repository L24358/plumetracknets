import os
import pickle
import numpy as np
import reducer.support.dynamics as dy
from scipy.integrate import odeint
from reducer.config import modelpath # TODO: fix this
from reducer.support.basics import single_sine, constant # TODO: fix this

specify, episode, T, rp = 0, 5, 128, 1000
training_data = dy.generate_single_trial(specify, episode, T, rp)
validation_data = dy.generate_single_trial(specify, episode, T, 200)

fname = os.path.join("/src/aesindyc/", 'experiment_results_202302010002.pkl')
with open(fname, 'rb') as f: res = pickle.load(f)

def build_rhs_poly3_wsine(y, u, coefs, mask, total_dim):

    y = np.append(y, u)

    def constant(c): return c

    def ijk(y, c, i, j, k): return c*y[i]*y[j]*y[k]
    
    def sine(y, c, i): return c*np.sin(y[i])

    rhs = []
    # for constant
    if mask[0]: rhs.append(constant(coefs[0])) 
    
    # for cross terms
    count = 1
    for i in range(total_dim):
        for j in range(i, total_dim):
            for k in range(j, total_dim):
                if mask[count]: rhs.append(ijk(y, coefs[count], i, j, k))
                count += 1

    # for sine terms
    for i in range(total_dim):
        if mask[count]: rhs.append(sine(y, coefs[count], i))
        count += 1

    return sum(rhs)

def rhs(y, t, u, coefs, mask, total_dim):
    dydt = [build_rhs_poly3_wsine(y, u(t), coefs[i], mask[i], total_dim) for i in range(len(coefs))]
    return np.array(dydt)

def get_u():
    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f: dic = pickle.load(f)
    def inner(t):
        C = dic["C"][1](t, *dic["C"][0])
        y = dic["y"][1](t, *dic["y"][0])
        x = dic["x"][1](t, *dic["x"][0])
        observations = np.vstack((C, y, x)).T
        return observations
    return inner

d = 3 # latent_dim
p = 3 # ctrl_dim
T = 1000
u = get_u()
t = np.arange(0, T, 1)
y0 = np.random.normal(0, 3, size=(d,))
coefs = res["sindy_coefficients"][0].T
mask = res["coefficient_mask"][0].T
sol = odeint(rhs, y0, t, args=(u, coefs, mask, d + p))

import reducer.support.visualization as vis
vis.plot_PCA_3d(sol, figname="temp.png", save=True, plot_time=True)