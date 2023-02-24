"""
What does a correct result look like?
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
episode = "random"

# load data
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
dic = bcs.simulation_loader(specify, tpe, episode=episode)
observations = dic["observations"]
actions = dic["actions"]

# define agent class for convenience
class Agent():
    def __init__(self):
        self.loc = np.zeros(2) # origin
        self.angle = np.pi # absolute angle; facing towards -y
        self.history = np.array([self.loc, self.angle])

    def update(self, r, theta):
        self.loc += r*np.array([np.cos(theta), np.sin(theta)])
        self.angle += theta
        self.log()

    def log(self):
        self.history = np.insert(self.history, -1, [self.loc, self.angle])

# get absolute wind angle
agent = Agent()
abs_wind_angles = []
for t in range(len(observations)):
    x, y, C = observations[t]
    r, theta = actions[t]

    ego_wind_angle = np.arctan2(y, x)
    abs_wind_angle = agent.angle + ego_wind_angle
    agent.update(r, theta)
    abs_wind_angles.append(abs_wind_angle % np.pi)

plt.hist(abs_wind_angles)
vis.savefig()