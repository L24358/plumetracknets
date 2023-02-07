import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import reducer.support.basics as bcs
from scipy.optimize import fsolve
from reducer.support.exceptions import AlgorithmError
from reducer.support.odesolver import Discrete
from reducer.config import modelpath

########################################################
#                  Fixed Point Related                 #
########################################################

def rhs(h, args):
    '''rhs - lhs of the RNN, i.e. tanh(...) - h = 0.'''
    x, rnn, inn, br, bi = args
    return np.tanh(rnn @ h + inn @ x + br + bi) - h

def jacobian(h, args):
    '''Jacobian of the RNN.'''
    x, rnn, inn, br, bi = args
    inner = rnn @ h + inn @ x  + br + bi
    return np.diag((1 - pow(np.tanh(inner), 2))) @ rnn

def jacobian_rhs(h, args):
    '''Jacobian of the rhs of the RNN.'''
    return jacobian(h, args) - np.identity(64)

def check_fixed_point(fp):
    '''Check if fixed point is between -1 and 1 (because of tanh).'''
    return np.all([abs(c) <= 1 for c in fp])

def get_fixed_points(x, rnn, inn, br, bi, rp=100):
    '''Obtain fixed points, with #rp random initial values.'''
    fps = []
    for _ in range(rp):
        h_0 = np.random.uniform(low=-1, high=1, size=64)
        fp, infodic, ier = fsolve(rhs, h_0, args=[x, rnn, inn, br, bi], xtol=1e-15, full_output=True)[:3]
        # fprime=jacobian_rhs

        flag = (abs(np.mean(infodic["fvec"])) <= 1e-15) and ier
        for ref in fps:
            if not bcs.different(fp, ref)[0]: flag = False; break
        if flag: fps.append(fp)

    if not np.all([check_fixed_point(fp) for fp in fps]): raise AlgorithmError("fixed point is not between -1 and 1!")
    return fps

def get_jacobians(x, rnn, inn, br, bi):
    '''Obtain Jacobian of all the fixed points.'''
    fps = get_fixed_points(x, rnn, inn, br, bi)
    if len(fps) > 1: print(f"{x} obtained {len(fps)} fixed points!")
    return [jacobian(fp, [x, rnn, inn, br, bi]) for fp in fps]

def get_stability(fp, args):
    J = jacobian(fp, args)
    evs = np.linalg.eigvals(J)
    stable = np.all(abs(evs) < 1)
    return stable

def get_sorted_eig(fp, args):
    J = jacobian(fp, args)
    evs, ews = np.linalg.eig(J)
    idx = np.flip(np.argsort(abs(evs)))
    evs_sorted = evs[idx]
    ews_sorted = ews.T[idx]
    return evs_sorted, ews_sorted # rows = eigenvectors

########################################################
#               Matrix Numerical Methods               #
########################################################

def low_rank_approximation(rnn, r):
    U, s, VT = np.linalg.svd(rnn)
    S = np.zeros(rnn.shape)
    np.fill_diagonal(S, s)
    rrnn = U[:,:r] @ S[:r,:r] @ VT[:r,:]
    return rrnn

########################################################
#                  Simulation Related                  #
########################################################

def polar_to_cartesian(actions, scale=np.pi):
    r, theta = np.array(actions).T
    x = r*np.cos(theta*scale)
    y = r*np.sin(theta*scale)
    return x, y

def cartesian_to_polar(coors):
    x, y = np.array(coors).T
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def get_rotation_matrix(coor): # clockwise?
    r, theta = cartesian_to_polar([coor])
    theta = theta.item()
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def get_trajectory(actions):
    '''
    Get (x, y) trajectory from actions.
    @ Args:
        - actions: np.ndarray, shape = (#T, 2)
    @ Returns:
        - coor: np.ndarray, shape = (#T, 2)
    '''
    x, y = polar_to_cartesian(actions)
    coor = np.zeros((1,2))
    for t in range(len(x)):
        R = get_rotation_matrix(coor[-1])
        v = np.array([x[t], y[t]])
        new_coor = coor[-1] + R @ v
        coor = np.vstack((coor, new_coor))
    return coor

def sim(Wh, Wc, br, bi, obs, h_0, **kwargs):
    '''
    Simulates the RNN.
    @ Returns:
        - rest: np.ndarray, time points, shape = (#T,)
        - resy: np.ndarray, variables, shape = (#T, #vars)
    '''
    kw = {"T": 100}
    kw.update(kwargs)

    def rhs(h, t): return np.tanh(Wh @ h + Wc @ obs(t) + br + bi)

    solver = Discrete(rhs)
    solver.set_initial_conditions(h_0)
    resy, rest = solver.solve(np.arange(0, kw["T"]))
    return rest, resy

def sim_actor(Wh, Wc, br, bi, obs, h_0, specify, **kwargs):
    '''
    Simulates the RNN plus the actor.
    @ Returns:
        - rest: np.ndarray, time points, shape = (#T,)
        - resy: np.ndarray, variables, shape = (#T, #vars)
        - actions: np.ndarray, actions, shape = (#T, 2)
    '''
    kw = {"T": 100}
    kw.update(kwargs)
    rest, resy = sim(Wh, Wc, br, bi, obs, h_0, **kw)

    dic = bcs.actor_loader(specify=specify)
    actor = Actor()
    actor.init_params(dic)
    actions = actor(resy).squeeze().detach().numpy()
    return rest, resy, actions

def constant_obs(x_0):
    '''Constant input to the RNN.'''
    def inner(t): return np.array(x_0)
    return inner

def assigned_obs(x):
    '''Assigned input to the RNN.'''
    def inner(t): return x[t]
    return inner

def single_sine_obs(params):
    '''Single sine wave input to the RNN.'''
    def inner(t):
        C = params["C"][1](t, *params["C"][0])
        y = params["y"][1](t, *params["y"][0])
        x = params["x"][1](t, *params["x"][0])
        return np.array([C, y, x])
    return inner

def generate_single_trial(specify, episode, T, rp):
    '''UNFINISHED.'''
    rnn, inn, br, bi = bcs.model_loader(specify=specify)
    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f: dic = pickle.load(f)
    t = np.arange(T+1)
    fitfunc = bcs.FitFuncs()
    C = fitfunc(dic["C"][1])(t, *dic["C"][0])
    y = fitfunc(dic["y"][1])(t, *dic["y"][0])
    x = fitfunc(dic["x"][1])(t, *dic["x"][0])
    observations = np.vstack((C, y, x)).T

    keys = ["C", "y", "x"]
    As = [dic[k][0][0] for k in keys]
    fs = [dic[k][0][1] for k in keys]
    phis = [dic[k][0][2] for k in keys]
    bs = [dic[k][0][3] for k in keys]

    y_rnns = []
    for _ in range(rp):
        h_0 = np.random.uniform(low=-1, high=1, size=(64,))
        _, y_rnn = sim(rnn, inn, br, bi, assigned_obs(observations), h_0, T=T+1)
        y_rnns.append(y_rnn[1:])
    y_rnns = np.vstack(y_rnns)

    t_tiled = np.tile(t, rp).reshape(-1, 1)
    u = np.tile(observations, (rp, 1))
    var = y_rnns
    dvar = var[1:] - var[:-1]
    dic = {"t": t_tiled, "x": var[1:], 'dx': dvar, "u": u}
    return dic

########################################################
#             Actor (code from satsingh)               #
########################################################

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor1 = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.dist = DiagGaussian(64, 2)

    def init_params(self, dic):
        self.actor1[0].weight = nn.Parameter(torch.from_numpy(dic["base.actor1.0.weight"]))
        self.actor1[0].bias = nn.Parameter(torch.from_numpy(dic["base.actor1.0.bias"]))
        self.actor[0].weight = nn.Parameter(torch.from_numpy(dic["base.actor.0.weight"]))
        self.actor[0].bias = nn.Parameter(torch.from_numpy(dic["base.actor.0.bias"]))
        self.dist.fc_mean.weight = nn.Parameter(torch.from_numpy(dic["dist.fc_mean.weight"]))
        self.dist.fc_mean.bias = nn.Parameter(torch.from_numpy(dic["dist.fc_mean.bias"]))
        self.dist.logstd._weight = nn.Parameter(torch.from_numpy(dic["dist.logstd._bias"]))

    def forward(self, x):
        if type(x) == np.ndarray: x = torch.from_numpy(x.astype(np.float32))
        h = self.actor1(x)
        y = self.actor(h)
        dist = self.dist(y)
        return dist.mode()

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module