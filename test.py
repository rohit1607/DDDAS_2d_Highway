# import numpy as np
# import random
# from utils.custom_functions import calc_mean_and_std_train_test, get_rzn_ids_for_training_and_testing


# train_id_list, test_id_list = get_rzn_ids_for_training_and_testing(40,50)
# scalar_list = [50 for i in range(40)]
# for i in range(10):
#     scalar_list.append(None)
# scalar_list = list(scalar_list)
# print(type(scalar_list))
# scalar_list[_train = None
# print("tr ", train_id_list)
# print("tes ",test_id_list)
# print("sc ",scalar_list)

# mean, std, cnt, none_cnt, none_cnt_perc = calc_mean_and_std_train_test(scalar_list, train_id_list, test_id_list)
# print(mean, std, cnt, none_cnt, none_cnt_perc)
# from os.path import join
# from utils.custom_functions import append_summary_to_summaryFile
# from definition import ROOT_DIR
# output_paramaters_ith_case = ["exp_num", "method", "num_actions", "nt", "dt", "F", "startpos", "endpos", "eps_0", "ALPHA",
#                                 "eps_dec_method", "N_inc", "dt_size", "with_guidance", "init_Q", "num_passes", "QL_Iters",
#                                 "avg_time_p1_train", "std_time_p1_train", "avg_G0_p1_train", "none_cnt_p1_train", "cnt_p1_train", "none_cnt_perc_p1_train", #train stats
#                                 "avg_time_p2_train", "std_time_p2_train", "avg_G0_p2_train", "none_cnt_p2_train", "cnt_p2_train", "none_cnt_perc_p2_train", #train stats
#                                 "avg_time_p1_test", "std_time_p1_test", "avg_G0_p1_test", "none_cnt_p1_test", "cnt_p1_test", "none_cnt_perc_p1_test", #test stats
#                                 "avg_time_p2_test", "std_time_p2_test", "avg_G0_p2_test", "none_cnt_p2_test", "cnt_p2_test", "none_cnt_perc_p2_test", #test stats                                            
#                                 "overall_bad_count_p1", "overall_bad_count_p2", "case_runtime" ]

# append_summary_to_summaryFile( join(ROOT_DIR, 'Experiments/Exp_summary_QL.csv'),  output_paramaters_ith_case)

from utils.setup_grid import setup_grid
from definition import ROOT_DIR
from os.path import join
from utils.custom_functions import read_pickled_File, max_dict, picklePolicy, calc_mean_and_std, writePolicytoFile
from QL.Build_Q_from_Trajs import Q_update
import matplotlib.pyplot as plt
import copy

def run_onboard_routing_for_test_data(exp_num_case_dir, setup_grid_params):
   
    global ALPHA
    global N_inc
    Q = read_pickled_File(join(exp_num_case_dir, 'Q2'))
    N = read_pickled_File(join(exp_num_case_dir, 'N2'))
    test_id_list = read_pickled_File(join(exp_num_case_dir, 'test_id_list'))
    train_id_list = read_pickled_File(join(exp_num_case_dir, 'train_id_list'))
    sars_traj_list = read_pickled_File(join(exp_num_case_dir, 'sars_traj_Trained_Trajectories_after_exp'))
    train_output_params = read_pickled_File(join(exp_num_case_dir, 'output_paramaters'))

    print("len(sars_traj_list) = ", len(sars_traj_list))
    for i in range(len(sars_traj_list)):
        sars_traj = sars_traj_list[i]
        print(sars_traj[0])
    return


setup_grid_params = setup_grid(num_actions=16, nt = 100)
rel_path = 'Experiments/72/QL/num_passes_50/QL_Iter_x1/dt_size_4800/ALPHA_0.05/eps_0_0.33'
exp_num_case_dir = join(ROOT_DIR, rel_path)

run_onboard_routing_for_test_data(exp_num_case_dir, setup_grid_params)