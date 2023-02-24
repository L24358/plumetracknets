import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
from scipy.optimize import fsolve

# load
specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# functions
def rhs(h, args):
    x, rnn, inn, br, bi = args
    return rnn @ h + inn @ x + br + bi - np.arctanh(h)

def get_fixed_points2(x, rnn, inn, br, bi, rp=100):
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

    if not np.all([dy.check_fixed_point(fp) for fp in fps]): raise bcs.AlgorithmError("fixed point is not between -1 and 1!")
    return fps

x = [0.3, 0.3, 1]
fps1 = dy.get_fixed_points(x, rnn, inn, br, bi, rp=100)
fps2 = get_fixed_points2(x, rnn, inn, br, bi, rp=100)

import pdb; pdb.set_trace()
