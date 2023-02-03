import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy

# Load model
specify = "random"
rnn, inn, br, bi = bcs.model_loader(specify=specify) 

# Fixed points for no input, no bias
x_0 = [0., 0., 0.]
args = [x_0, rnn, inn, np.zeros(64), np.zeros(64)]
fps = dy.get_fixed_points(*args)

# Fixed points for no input, with bias
x_0 = [0, 0, 0]
args = [x_0, rnn, inn, br, bi]
fps = dy.get_fixed_points(*args)

# Fixed points with input and bias
x_0 = [1, 1, 1]
args = [x_0, rnn, inn, br, bi]
fps = dy.get_fixed_points(*args)