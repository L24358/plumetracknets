import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
from aesindy.example_lorenz import get_lorenz_data
from aesindy.sindy_utils import library_size
from aesindy.training import train_network
import tensorflow.compat.v1 as tf

# generate training, validation, testing data
noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength)
validation_data = get_lorenz_data(20, noise_strength=noise_strength)

print(training_data.keys())
print(training_data["x"].shape)
print(training_data["dx"].shape)