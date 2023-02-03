import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy

# parameters
specify = 0
episode = 5

# Load model
rnn, inn, br, bi = bcs.model_loader(specify=specify) 

# Fixed points for no input, no bias
x_0 = [0, 0, 0]
args = [x_0, rnn, inn, np.zeros(br.shape), np.zeros(bi.shape)]
fps = dy.get_fixed_points(*args)

# Fixed points for no input, with bias
x_0 = [0, 0, 0]
args = [x_0, rnn, inn, br, bi]
fps = dy.get_fixed_points(*args)

# Fixed points with input and bias
x_0 = [1, 1, 1]
args = [x_0, rnn, inn, br, bi]
fps = dy.get_fixed_points(*args)