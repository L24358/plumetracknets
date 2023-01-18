import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.navigator as nav
from reducer.config import modelpath

# Extract RNN matrices
if 0:
    modelnames = nav.model_finder()
    for modelname in modelnames:
        print(modelname)
        os.system(f"python3 /src/tracer/ppo/evalCli.py --model_fname {modelname}")
        # TODO: Not saving in the right folder

matrix = np.load(os.path.join(modelpath, "rnn_matrix", "base.rnn.weight_hh_l0.npy"))
rnn1, rnn2, rnn3 = matrix[:64], matrix[64:-64], matrix[-64:]

ev1, ew1 = np.linalg.eig(rnn1)
ev2, ew2 = np.linalg.eig(rnn2)
ev3, ew3 = np.linalg.eig(rnn3)

fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_subplot(131)
ax1.plot(sorted(ev1, reverse=True), 'k')
ax1.set_title("rnn1"); ax1.set_xlabel("eigenvalues")
ax2 = fig.add_subplot(132)
ax2.plot(sorted(ev2, reverse=True), 'k')
ax2.set_title("rnn2"); ax2.set_xlabel("eigenvalues")
ax3 = fig.add_subplot(133)
ax3.plot(sorted(ev3, reverse=True), 'k')
ax3.set_title("rnn3"); ax3.set_xlabel("eigenvalues")

plt.tight_layout()
plt.savefig("/src/reducer/graphs/temp.png", dpi=200)
