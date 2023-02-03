import os
import pickle
import numpy as np
import reducer.support.dynamics as dy
from scipy.integrate import odeint
from reducer.config import modelpath

def build_rhs_poly3_nsine(y, u, coefs, mask, total_dim):

    y = np.append(y, u)

    def constant(c): return c

    def ijk(y, c, i, j, k): return c*y[i]*y[j]*y[k]

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

    return sum(rhs)

def rhs(y, t, u, coefs, mask, total_dim):
    dydt = [build_rhs_poly3_nsine(y, u(t), coefs[i], mask[i], total_dim) for i in range(len(coefs))]
    return np.array(dydt)

fname = os.path.join(modelpath, "aesindy_dump", 'experiment_results_202302012324.pkl')
with open(fname, 'rb') as f: res = pickle.load(f)

d = 3 # latent_dim
p = 1 # ctrl_dim
T = 1000
u = lambda t: 0*np.sin(10*t)
t = np.arange(0, T, 1)
y0 = np.random.normal(0, 10, size=(d,))
coefs = res["sindy_coefficients"][0].T
mask = res["coefficient_mask"][0].T
cmask = np.multiply(coefs, mask)
sol = odeint(rhs, y0, t, args=(u, coefs, mask, d + p))

import reducer.support.visualization as vis
vis.plot_PCA_3d(sol, figname="temp.png", save=True, plot_time=True)

import pdb; pdb.set_trace()