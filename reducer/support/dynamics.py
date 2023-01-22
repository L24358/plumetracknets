import numpy as np
import reducer.support.basics as bcs
from scipy.optimize import fsolve
from reducer.support.exceptions import AlgorithmError
from reducer.support.odesolver import Discrete

def rhs(h, args):
    x, rnn, inn, br, bi = args
    return np.tanh(rnn @ h + inn @ x + br + bi) - h

def jacobian(h, args):
    x, rnn, inn, br, bi = args
    inner = rnn @ h + inn @ x  + br + bi
    return np.diag((1 - pow(np.tanh(inner), 2))) @ rnn

def jacobian_rhs(h, args):
    return jacobian(h, args) - np.identity(64)

def check_fixed_point(fp): return np.all([abs(c) <= 1 for c in fp])

def get_fixed_points(x, rnn, inn, br, bi, rp=100):
    fps = []
    for _ in range(rp):
        h_0 = np.random.uniform(low=-1, high=1, size=64)
        fp, infodic, ier = fsolve(rhs, h_0, args=[x, rnn, inn, br, bi], fprime=jacobian_rhs, xtol=1e-15, full_output=True)[:3]

        flag = (abs(np.mean(infodic["fvec"])) <= 1e-7) and ier
        for ref in fps:
            if not bcs.different(fp, ref)[0]: flag = False; break
        if flag: fps.append(fp)

    if not np.all([check_fixed_point(fp) for fp in fps]): raise AlgorithmError("fixed point is not between -1 and 1!")
    return fps

def get_jacobians(x, rnn, inn, br, bi):
    fps = get_fixed_points(x, rnn, inn, br, bi)
    if len(fps) > 1: print(f"{x} obtained {len(fps)} fixed points!")
    return [jacobian(fp, x, rnn, inn, br, bi) for fp in fps]

def sim(Wh, Wc, br, bi, obs, h_0, **kwargs):
    kw = {"T": 100}
    kw.update(kwargs)

    def rhs(h, t): return np.tanh(Wh @ h + Wc @ obs(t) + br + bi)

    solver = Discrete(rhs)
    solver.set_initial_conditions(h_0)
    resy, rest = solver.solve(np.arange(0, kw["T"]))
    return rest, resy

def constant_obs(x_0):
    def inner(t): return np.array(x_0)
    return inner