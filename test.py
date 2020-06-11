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
