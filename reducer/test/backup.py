import numpy as np
import reducer.support.basics as bcs
import reducer.support.visualization as vis

specify = 0
episode = np.random.choice(240)

sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
observations = sim_results["observations"]

traj = []
for i in range(3): traj.append([range(len(observations)), observations[:,i]])
vis.plot_multiple_trajectory(traj,
                            figname=f"observation_agent={specify+1}_tpe=constant_episode={episode}.png",
                            plot_time=False,
                            xlabel=["time"]*3,
                            ylabel=["value"]*3,
                            subtitle=["Concentration", "y", "x"],
                            suptitle=f"Agent {specify+1}, constant, episode {episode}")

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)

color = ["k", "r", "b"]
label = ["Concentration", "y", "x"]
xlabel = ["time", "Concentration"]
ylabel = ["value", "y"]
subtitle = ["", ""]
suptitle = f"Agent {specify+1}, constant, episode {episode}"
for i in range(3):
    ax1.plot(observations[:,i], color=color[i], label=label[i])
    ax1.legend(); ax1.set_xlabel(xlabel[0]); ax1.set_ylabel(ylabel[0]); ax1.set_title(subtitle[0])

ax2 = fig.add_subplot(122)
ax2.scatter(observations[:,0], observations[:,1], color="k")
ax2.annotate("$r^2$ = {:.3f}".format(r2_score(observations[:,0], observations[:,1])), (0, 1))
ax2.set_xlabel(xlabel[1]); ax2.set_ylabel(ylabel[1]); ax2.set_title(subtitle[1])

plt.suptitle(suptitle)
plt.tight_layout()
plt.savefig(f"/src/reducer/graphs/C-y_agent={specify+1}_tpe=constant_episode={episode}.png")

def eval_fit(target):
    dic = {}
    funcs = [single_sine, constant, envelope_sine]
    for func in funcs:
        res = differential_evolution(MSE, bounds(func), args=(target, func))
        error = MSE(res.x, target, func)
        dic[error] = [res.x, func]
    err_ss, err_c, err_es = list(dic.keys())
    if err_es < err_ss/1.1: minn = err_es # discount factor for envelope
    elif err_ss < err_c/1.5: minn = err_ss # discount factor for constant
    else: minn = err_c 

data = {(single_sine, "C"): [], (single_sine, "y"): [], (single_sine, "x"): [],
        (constant, "C"): [], (constant, "y"): [], (constant, "x"): []}

keys = ["C", "y", "x"]
for episode in episodes:
    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f:
        temp = pickle.load(f)
        
        for key in keys:
            data[(temp[key][1], key)].append(temp[key][0])

for key in keys:
    counts = np.expand_dims(np.asarray(data[(single_sine, key)]).T, axis=0)
    vis.plot_multiple_hist2(counts, figname="temp.png", subtitle=["A","f","phi","b","s"], ylabel=["count"]*5, title=key)

