'''
@ Things to try:
    - Should try discrete_time = True in SINDy()

@ references:
    - https://github.com/dynamicslab/pysindy
'''

from pysindy import SINDy # TODO: rebuild container
from aesindyc.generator import get_lorenz_data # TODO: include in setup.py

# generate data
A = 0
noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength, A=A) # u.shape = (T, ctrl_dim)
validation_data = get_lorenz_data(20, noise_strength=noise_strength, A=A)

# define and fit model
model = SINDy()
model.fit(training_data["x"], u=training_data["u"])
eqs = model.equations()