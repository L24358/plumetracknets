import os
import numpy as np
import pickle
import reducer.support.dynamics as dy
from scipy.integrate import odeint
from reducer.config import modelpath

specify, episode, T, rp = 0, 5, 128, 1000
training_data = dy.generate_single_trial(specify, episode, T, rp)
validation_data = dy.generate_single_trial(specify, episode, T, 200)

fname = os.path.join(modelpath, "aesindy_dump", 'experiment_results_202302010002.pkl')
with open(fname, 'rb') as f: res = pickle.load(f)

def build_rhs_poly3_nsine(y, u, coefs, mask, total_dim):

    y = np.append(y, u)

    def constant(c): return c

    rhs = []
    # for constant
    if mask[0]: rhs.append(constant(coefs[0])) 
    
    # for cross terms, 1
    count = 1
    for i in range(total_dim):
        if mask[count]: rhs.append(coefs[count]*y[i])
        count += 1

    for i in range(total_dim):
        for j in range(i, total_dim):
            if mask[count]: rhs.append(coefs[count]*y[i]*y[j])
            count += 1

    for i in range(total_dim):
        for j in range(i, total_dim):
            for k in range(j, total_dim):
                if mask[count]: rhs.append(coefs[count]*y[i]*y[j]*y[k])
                count += 1

    return sum(rhs)

def rhs(y, t, u, coefs, mask, total_dim):
    dydt = [build_rhs_poly3_nsine(y, u(t), coefs[i], mask[i], total_dim) for i in range(len(coefs))]
    return np.array(dydt)

print("Please adjust the parameters p and u here before the simulation!")
d = 3 # latent_dim
p = 3 # ctrl_dim
u = lambda t: training_data["u"][int(t)]
t = np.arange(1000)
y0 = np.random.uniform(low=-1, high=1, size=(d,))
coefs = res["sindy_coefficients"][0].T
mask = res["coefficient_mask"][0].T
cmask = np.multiply(coefs, mask)
sol = odeint(rhs, y0, t, args=(u, coefs, mask, d + p))

import reducer.support.visualization as vis
vis.plot_PCA_3d(sol[50:], figname="temp.png", save=True, plot_time=True)

import pdb; pdb.set_trace()