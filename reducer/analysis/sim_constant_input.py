import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy

dic = bcs.actor_loader(specify=0)
actor = dy.Actor()
actor.init_params(dic)
temp = np.random.normal(size=64)
action = actor(temp).squeeze()[0].detach().numpy()
