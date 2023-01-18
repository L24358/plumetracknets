"""
Adapted from evalCli.py.
"""

from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import traceback

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pprint import pprint
import glob
sys.path.append('/home/satsingh/plume/plume2/')
from tracer.plume_env import PlumeEnvironment, PlumeFrameStackEnvironment
import tracer.agent_analysis as agent_analysis
import tracer.log_analysis as log_analysis
from tracer.ppo.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from tracer.ppo.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from reducer.config import modelpath

parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--algo', default='ppo')
parser.add_argument('--dataset', default='constantx5b5')
parser.add_argument('--model_fname')
parser.add_argument('--test_episodes', type=int, default=100)
parser.add_argument('--viz_episodes', type=int, default=10)
parser.add_argument('--fixed_eval', action='store_true', default=False)
parser.add_argument('--test_sparsity', action='store_true', default=False)
parser.add_argument('--diffusionx',  type=float, default=1.0) # env related

args = parser.parse_args()
print(args)
args.det = True # override

np.random.seed(args.seed)
args.env_name = 'plume'
args.env_dt = 0.04
args.turnx = 1.0
args.movex = 1.0
args.birthx = 1.0
args.loc_algo = 'quantile'
args.time_algo = 'uniform'
args.diff_max = 0.8
args.diff_min = 0.8
args.auto_movex = False
args.auto_reward = False
args.wind_rel = True
args.action_feedback = False
args.walking = False
args.radiusx = 1.0
args.r_shaping = ['step'] # redundant
args.rewardx = 1.0
args.squash_action = True
args.diffusion_min = args.diffusionx
args.diffusion_max = args.diffusionx
args.flipping = False
args.odor_scaling = False
args.qvar = 0.0
args.stray_max = 2.0
args.birthx_max = 1.0
args.masking = None
args.stride = 1
args.obs_noise = 0.0
args.act_noise = 0.0
args.dynamic = False
args.recurrent_policy = True if ('GRU' in args.model_fname) or ('RNN' in args.model_fname) else False
args.rnn_type = 'VRNN' if 'RNN' in args.model_fname else 'GRU'
args.stacking = 0
if 'MLP' in args.model_fname:
    args.stacking = int( args.model_fname.split('MLP_s')[-1].split('_')[0] )

# load trained model
actor_critic, ob_rms = torch.load(args.model_fname, map_location=torch.device('cpu'))

# save RNN parameters
for name, param in actor_critic.named_parameters():
    if "rnn" in name:
        np.save(os.path.join(modelpath, "rnn_matrix", name + ".npy"), param.detach().numpy())
