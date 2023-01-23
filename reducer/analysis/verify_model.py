import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

specify = 2
rp = 5
trajs = []
for _ in range(rp):
    sim_results = bcs.simulation_loader(specify, "constant", episode="random")
    observations = sim_results["observations"][:]
    actions = sim_results["actions"][:]
    h_0 = sim_results["activities_rnn"][0]
    observations = bcs.upsample(observations, 1)

    rnn, inn, br, bi = bcs.model_loader(specify=specify)
    t, y_rnn, actions_rnn = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, specify, T=len(observations))

    traj = dy.get_trajectory(actions)
    traj_rnn = dy.get_trajectory(actions_rnn)
    trajs.append([traj.T, traj_rnn.T])

fig = vis.plot_multiple_trajectory2(trajs, save=False)
vis.common_col_title(fig, ["original", "new"], [rp, 2])
vis.savefig(figname="verify_model3.png")