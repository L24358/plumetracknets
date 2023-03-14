import os
import pickle

fname = os.path.join("/src/reducer/test/", 'experiment_results_202301302031.pkl')
with open(fname, 'rb') as f:
    res = pickle.load(f)

import numpy as np
np.multiply(res["sindy_coefficients"][0], res["coefficient_mask"][0])

import pdb; pdb.set_trace()