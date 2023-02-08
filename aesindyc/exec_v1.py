import sys
sys.path.append("../../src")
import os
import pickle
import datetime
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
from aesindyc.sindy_utils import library_size
from aesindyc.training import train_network
from reducer.config import modelpath

params = {}

# sys.argv
latent_dim = int(sys.argv[1])
poly_order = int(sys.argv[2])
include_sine = bool(sys.argv[3])
coefficient_threshold = float(sys.argv[4])
threshold_frequency = int(sys.argv[5])
learning_rate = float(sys.argv[6])
epochs1, epochs2 = int(sys.argv[7]), int(sys.argv[8])
specify = int(sys.argv[9])
episode = int(sys.argv[10])
T, rp = int(sys.argv[11]), int(sys.argv[12])
foldername = sys.argv[13]

# load ptn data
training_data = dy.ptn_loader(specify, episode, rp, T)
validation_data = dy.ptn_loader(specify, episode, 200, T)

# important hyperparameters
params['ctrl_dim'] = 3
params['input_dim'] = 64
params['latent_dim'] = latent_dim
params['model_order'] = 1
params['poly_order'] = poly_order
params['include_sine'] = include_sine
params['library_dim'] = library_size(params['latent_dim'] + params['ctrl_dim'], params['poly_order'], params['include_sine'], True)

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = coefficient_threshold
params['threshold_frequency'] = threshold_frequency
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 0.0
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_regularization'] = 1e-5

params['activation'] = 'sigmoid'
params['widths'] = [64,32]

# training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = T
params['learning_rate'] = learning_rate

params['data_path'] = os.path.join(modelpath, foldername)
params['print_progress'] = True
params['print_frequency'] = 100

# training time cutoffs
params['max_epochs'] = epochs1
params['refinement_epochs'] = epochs2

# save
info = {
    "specify": specify,
    "episode": episode,
    "T": T,
    "rp": rp,
    "latent_dim": latent_dim,
    "poly_order": poly_order,
    "include_sine": include_sine,
    "coefficient_threshold": coefficient_threshold,
    "threshold_frequency": threshold_frequency,
    "learning_rate": learning_rate,
    "epochs1": epochs1,
    "epochs2": epochs2,
    "seed_train": training_data["seed"],
    "seed_validation": validation_data["seed"],
    "clip": training_data["clip"],
    "noise_std": training_data["noise_std"],
    "fit_dic": training_data["fit_dic"],
}
id = np.random.randint(0,99999)
bcs.dump(info, foldername, f"ptn_id={id}.pkl")

# run
num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = f'ptn_id={id}' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

df.to_pickle(os.path.join(params['data_path'], f'results_id={id}.pkl'))
print("DONE.")

