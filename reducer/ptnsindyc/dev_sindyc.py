'''
@ Things to try:
    - Should try discrete_time = True in SINDy()

@ references:
    - https://github.com/dynamicslab/pysindy
'''

import numpy as np
from scipy.integrate import odeint
from pysindy import SINDy # TODO: rebuild container
from aesindyc.generator import get_lorenz_data # TODO: include in setup.py

# parameters
A = 0

# generate data
u = lambda t : np.sin(2 * t) ** 2
lorenz_c = lambda z,t : [
                10 * (z[1] - z[0]) + u(t),
                z[0] * (28 - z[2]) - z[1],
                z[0] * z[1] - 8 / 3 * z[2],
        ]

t = np.arange(0,2,0.002)
x = odeint(lorenz_c, [-8,8,27], t)
u_eval = u(t)
training_data = {"x": x, "u": u_eval, "t": t[1]-t[0]}

# define and fit model
model = SINDy()
model.fit(training_data["x"], u=training_data["u"], t=training_data["t"])
model.print()
import pdb; pdb.set_trace()