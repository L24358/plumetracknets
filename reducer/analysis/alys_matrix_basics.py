import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.navigator as nav
from reducer.config import modelpath, graphpath

# Extract (save as npy) RNN matrices
if 0:
    modelnames = nav.file_finder()
    for modelname in modelnames:
        print(modelname)
        os.system(f"python3 /src/tracer/ppo/evalCli.py --model_fname {modelname}")

# Eigenvalue spectrum of recurrent matrix
if 1:
    modelnames = nav.file_finder(
        target="weight_hh",
        extension=".npy",
        parent_name=os.path.join(modelpath, "rnn_matrix"))

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)

    for modelname in modelnames:
        rnn = np.load(modelname)
        ev, ew = np.linalg.eig(rnn)
        
        filename = os.path.splitext(os.path.basename(modelname))[0]
        seed = bcs.param_finder(filename, "seed", sep2=None)

        ax1.plot(sorted(ev, reverse=True), label=seed)
        
    ax1.set_title("rnn"); ax1.set_ylabel("eigenvalues")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphpath, "eigenvalues.png"), dpi=200)
    plt.clf()

# Connection weights of the input matrix
if 1:
    modelnames = nav.file_finder(
        target="weight_ih",
        extension=".npy",
        parent_name=os.path.join(modelpath, "rnn_matrix"))

    for modelname in modelnames:
        rnn = np.load(modelname)
        filename = os.path.splitext(os.path.basename(modelname))[0]
        seed = bcs.param_finder(filename, "seed", sep2=None)

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121)
        sns.heatmap(rnn, ax=ax1)
        ax1.set_xlabel("input variables"); ax1.set_ylabel("units")
        ax1.set_title("Input Connection Matrix")

        ax2 = fig.add_subplot(122)
        colors = ["b", "g", "r"]
        for i in range(3):
            ax2.plot(sorted(rnn[:,i], reverse=True), color=colors[i], label=f"Input {i}")
        ax2.set_xlabel("sorted values"); ax2.set_ylabel("connection weight")
        ax2.set_title("Connection Weight Distribution")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graphpath, f"connection_weights_seed={seed[4:]}.png"), dpi=200)
        plt.clf()

# Hierarchical clustering of recurrent matrix
if 1:
    modelnames = nav.file_finder(
        target="weight_hh",
        extension=".npy",
        parent_name=os.path.join(modelpath, "rnn_matrix"))

    for modelname in modelnames:
        filename = os.path.splitext(os.path.basename(modelname))[0]
        seed = bcs.param_finder(filename, "seed", sep2=None)
        rnn = np.load(modelname)

        sns.clustermap(rnn)
        plt.title("Connectivity Matrix Clustermap")
        plt.savefig(os.path.join(graphpath, f"clustermap_seed={seed[4:]}.png"), dpi=200)
        plt.clf()