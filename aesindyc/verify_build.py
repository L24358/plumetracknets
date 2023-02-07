import os
import pickle
import numpy as np
import reducer.support.dynamics as dy
from scipy.integrate import odeint
from reducer.config import modelpath

def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):

    import tensorflow.compat.v1 as tf
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)

# sindy_library_tf([1,2,3], 3, 3, include_sine=False)

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

    return rhs

# 1, x1, x2, x3, x1^2, x1x2, x1x3, x2^2, x2x3, x3^2, x1^3, x1x1x2, x1x1x3, x1x2x2, ..
def rhs(y, t, u, coefs, mask, total_dim):
    dydt = [build_rhs_poly3_nsine(y, u(t), coefs[i], mask[i], total_dim) for i in range(len(coefs))]
    return np.array(dydt)

u = lambda t: 0
y = np.arange(1,4)
coefs = np.ones((3,20))
mask = np.where(coefs != 0, 1, 0)
res = build_rhs_poly3_nsine(y, u, coefs[0], mask[0], 3)

import pdb; pdb.set_trace()