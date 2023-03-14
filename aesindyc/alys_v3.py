import sys
sys.path.append("../../src")
import os
import pickle
import datetime
import pandas as pd
import numpy as np
from aesindy.example_lorenz import get_lorenz_data
from aesindy.sindy_utils import library_size
from aesindy.training import train_network
import tensorflow.compat.v1 as tf
from reducer.config import modelpath

# generate training, validation, testing data
noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength)
validation_data = get_lorenz_data(20, noise_strength=noise_strength)

fname = os.path.join(modelpath, "aesindy_dump", 'experiment_results_202302030414.pkl')
with open(fname, 'rb') as f: res = pickle.load(f)

coefs = res["sindy_coefficients"][0].T
mask = res["coefficient_mask"][0].T
cmask = np.multiply(coefs, mask)

print(cmask.T)
print(training_data["sindy_coefficients"])