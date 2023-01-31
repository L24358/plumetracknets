import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
import reducer.support.dynamics as dy
from aesindy.example_lorenz import get_lorenz_data # TODO: replace
from aesindyc.sindy_utils import library_size
from aesindyc.training import train_network
import tensorflow.compat.v1 as tf

params = {}

# generate training, validation, testing data
noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength)
validation_data = get_lorenz_data(20, noise_strength=noise_strength)

# NEW: ptn data
specify, episode, T, rp = 0, 5, 128, 1000
training_data = dy.generate_single_trial(specify, episode, T, rp)
validation_data = dy.generate_single_trial(specify, episode, T, 200)

# NEW: temporarily generate u(t) # TODO: fix
def gen_temp_u(N):
    t = np.arange(N)
    u = [np.sin(t), np.sin(2*t), np.sin(3*t)]
    u = np.vstack(u).T
    return u
training_data['u'] = gen_temp_u(training_data['x'].shape[0])
validation_data['u'] = gen_temp_u(validation_data['x'].shape[0])

# NEW: control dimensions
params['ctrl_dim'] = 3

params['input_dim'] = 64 # NEW
params['latent_dim'] = 3
params['model_order'] = 1
params['poly_order'] = 3
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'] + params['ctrl_dim'], params['poly_order'], params['include_sine'], True) # NEW

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 500
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
params['batch_size'] = 1024
params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 100

# training time cutoffs
params['max_epochs'] = 1001
params['refinement_epochs'] = 1001

num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = 'lorenz_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')