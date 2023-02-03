import os
import pickle

fname = os.path.join("/src/aesindyc/", 'experiment_results_202302010002.pkl')
with open(fname, 'rb') as f:
    res = pickle.load(f)

import numpy as np
mask = res["coefficient_mask"][0]
coef = res["sindy_coefficients"][0]
rhs = np.multiply(coef, mask)

print((rhs.T*1000).round())

import pdb; pdb.set_trace()