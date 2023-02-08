import os 
import numpy as np
import reducer.support.basics as bcs
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
episode = 95 #"random"

# load data
dic = bcs.simulation_loader(specify, tpe, episode=episode)
vis.plot_obs_act_traj(dic["actions"], dic["observations"], figname="temp.png")

