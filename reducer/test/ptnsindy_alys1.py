import os
import pickle

fname = os.path.join("/src/reducer/test/", 'experiment_results_202301300646.pkl')
with open(fname, 'rb') as f:
    res = pickle.load(f)

import pdb; pdb.set_trace()