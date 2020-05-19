# from utils.plot_functions import plot_paths_colored_by_EAT
#
# plotfile = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/7/QL/dt_size_5000/ALPHA_0.5/eps_0_0.25/Trajectories_after_exp'
# save_path = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/7/QL/dt_size_5000/ALPHA_0.5/eps_0_0.25/'
# save_fname = 'colored_EAT'
# full_name = save_path + save_fname
# time_list = plot_paths_colored_by_EAT(plotFile = plotfile, savePath_fname=full_name)

# from utils.build_model_GPU import get_S_from_S_id

# gsize = 100
# print(get_S_from_S_id(2e4 + 3e2 + 47, gsize))
# print(get_S_from_S_id(2e4, gsize))
# print(get_S_from_S_id(3e2 + 47, gsize))
# print(get_S_from_S_id(47, gsize))

# from utils.setup_grid import setup_grid
# Nt = 40
# g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, setup_params, setup_param_str = setup_grid(num_actions = 16, nt = Nt)

# ac_states = g.ac_state_space()

# for s in ac_states:
#     if s[0] == Nt -2:
#         print(s)


from utils.custom_functions import picklePolicy, calc_mean_and_std, read_pickled_File
fname ='/home/rohit/Documents/Research/ICRA_2020/DDDAS_2D_Highway/DP/Trans_matxs_3D/GPU_test_5_3D_90nT_a16/GPU_test_5_3D_90nT_a16'
a = read_pickled_File(fname)
# a[(89,1,1)]

for k in a:
    if k[0] == 89:
        print(k, a[k])
