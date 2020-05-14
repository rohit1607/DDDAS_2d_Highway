from utils.plot_functions import plot_paths_colored_by_EAT

plotfile = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/7/QL/dt_size_5000/ALPHA_0.5/eps_0_0.25/Trajectories_after_exp'
basefile = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/23/DP/Traj_set_'
save_path = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/7/QL/dt_size_5000/ALPHA_0.5/eps_0_0.25/'
save_fname = 'colored_EAT_QLvsDP'
full_name = save_path + save_fname
time_list = plot_paths_colored_by_EAT(plotFile = plotfile, baseFile= basefile, savePath_fname=full_name)