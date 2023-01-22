import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# Load model
rnn, inn, br, bi = bcs.model_loader(specify="random") # Take the first model
br = bi = np.zeros(br.shape) ##

# Obtain the fixed points
x_0 = [0, 0, 0]
args = [x_0, rnn, inn, br, bi]
fps = dy.get_fixed_points(*args)
Js = [dy.jacobian(fp, args) for fp in fps]

# Check how accurate the fixed point solutions are, and stability
for i in range(len(fps)):
    print(f"sum(rhs) for fp{i}: ", sum(dy.rhs(fps[i], args)))
    evs = np.linalg.eigvals(Js[i])
    print(f"Is the fixed point stable? ", np.all(abs(evs) < 1))

# Simulate and plot trial with fixed point as i.c.
h_0 = fps[0] # Take the first fixed point
t, y = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=100)
vis.plot_PCA_3d(y) # Plot first 3 PCA dimensions
