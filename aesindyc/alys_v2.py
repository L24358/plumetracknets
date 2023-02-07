import os
import pickle
import numpy as np
import reducer.support.dynamics as dy
from scipy.integrate import odeint
from reducer.config import modelpath

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

fname = os.path.join(modelpath, "aesindy_dump", 'experiment_results_202302052213.pkl')
with open(fname, 'rb') as f: res = pickle.load(f)

print("Please adjust the parameters p and u here before the simulation!")
d = 3 # latent_dim
p = 1 # ctrl_dim
u = lambda t: 10*np.sin(10*t)
t = np.arange(0,100,0.002)
y0 = [-8,8,27]
coefs = res["sindy_coefficients"][0].T
mask = res["coefficient_mask"][0].T
cmask = np.multiply(coefs, mask)
sol = odeint(rhs, y0, t, args=(u, coefs, mask, d + p))

import reducer.support.visualization as vis
vis.plot_PCA_3d(sol[50:], figname="temp.png", save=True, plot_time=True)

# from aesindy.example_lorenz import get_lorenz_data
# noise_strength = 1e-6
# training_data = get_lorenz_data(1024, noise_strength=noise_strength)
# coefs = training_data["sindy_coefficients"].T
# mask = np.where(coefs != 0, 1, 0)
# sol = odeint(rhs, y0, t, args=(u, coefs, mask, d + p))
# vis.plot_PCA_3d(sol, figname="temp.png", save=True, plot_time=True)

print(cmask)
import pdb; pdb.set_trace()