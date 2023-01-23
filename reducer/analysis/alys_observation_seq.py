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
