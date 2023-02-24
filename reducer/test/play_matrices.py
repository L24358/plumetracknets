import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from scipy.optimize import fsolve

# load
specify = 0
tpe = "constant"
plot = False
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# an hstar to play with
x = [-0.7608967,   0.70788926,  0.        ]
hstar = dy.get_fixed_points(x, rnn, inn, br, bi, rp=100)[0]
args = [x, rnn, inn, br, bi]

# get all obs
if 1:
    obs = []
    for episode in range(240):
        dic = bcs.simulation_loader(specify, tpe, episode=episode)
        obs.append(dic["observations"])
    obs = np.vstack(obs)

# find > 1 fps
if 0:
    for o in obs:
        hstars = dy.get_fixed_points(o, rnn, inn, br, bi, rp=100)
        if len(hstars) > 1:
            pass #print(o)

# play with
if 0:
    x = [-0.7608967,   0.70788926,  0.        ]
    hstar = dy.get_fixed_points(x, rnn, inn, br, bi, rp=100)[0]
    constant = inn @ x + br + bi
    recurrent = rnn @ hstar

    all_others = []
    for i in range(64):
        slope = rnn[i][i]
        others = recurrent[i] - slope*hstar[i]

        all_others.append(others)
        if plot:
            grid = np.linspace(-1+1e-04, 1-1e-04, 20)
            plt.plot(grid, slope*grid + others + constant[i])
            plt.plot(grid, np.arctanh(grid))
            vis.savefig()

    print("rnn_ij std: ", np.std(rnn.flatten()))
    print("Others std: ", np.std(all_others))
    print("hstar std: ", np.std(hstar))
    print("Means of rnn, others, hstar: ", np.mean(rnn.flatten()), np.mean(all_others), np.mean(hstar))

# create own matrix
std = np.std(rnn.flatten())
if 1:
    rnn_crazy = np.random.normal(0, std, size=(64,64))
    rnn_crazy = np.clip(-0.4, 0.4, rnn_crazy)
    rnn_crazy = rnn
    for o in obs:
        hstars = dy.get_fixed_points(o, rnn_crazy, inn, br, bi, rp=100)
        print(hstars)

if 0:
    J = dy.jacobian(hstar, args)
    import pdb; pdb.set_trace()