from QL.Build_Q_from_Trajs import Learn_policy_from_data
from utils.plot_functions import plot_max_delQs, plot_exact_trajectory_set, plot_max_Qvalues, plot_learned_policy
from QL.Q_Learning import Q_learning_Iters
from utils.custom_functions import initialise_policy, initialise_Q_N, initialise_guided_Q_N, writePolicytoFile, initialise_policy_from_initQ, createFolder, calc_mean_and_std, append_summary_to_summaryFile
import time
import math
import numpy as np
from definition import ROOT_DIR, N_inc
from os.path import join
Pi = math.pi


def run_QL(setup_grid_params, QL_params, QL_path, exp_num):
    
    exp =  QL_path

# init_Q = -1000000
#     stream_speed = 0.2

    Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, init_Q, with_guidance, method, num_passes, eps_dec_method = QL_params
    
    #Read data from files
    #setup_params (from setup_grid.py)= [num_actions, nt, dt, F, startpos, endpos]
    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, paths, setup_params, setup_param_str = setup_grid_params
    print("In TQLearn: ", len(setup_params), setup_params)
    num_actions, nt, dt, F, startpos, endpos = setup_params
    

    #print QL Parameters to file
    total_cases = len(Training_traj_size_list)*len(ALPHA_list)*len(esp0_list)
    str_Params = ['with_guidance','Training_traj_size_list', 'ALPHA_list', 'esp0_list', 'QL_Iters', 'num_actions', 'init_Q', 'dt', 'F']
    Params = [with_guidance, Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, num_actions, init_Q, dt, F]
    Param_filename = exp +'/Parmams.txt'
    outputfile = open(Param_filename, 'w+')
    for i in range(len(Params)):
        print(str_Params[i]+':  ',Params[i], file=outputfile)
    outputfile.close()
    

    #Create Sub-directories for different hyper parameters
    for eps_0 in esp0_list:
        for ALPHA in ALPHA_list:
            for dt_size in Training_traj_size_list:
                directory = exp + '/dt_size_' + str(dt_size) + '/ALPHA_' + str(ALPHA) + '/eps_0_' + str(eps_0)
                createFolder(directory)


    case =0 #initilise case. Each case is an experiment with a particular combination of eps_0, ALPHA and dt_size
    output_parameters_all_cases = []    # contains output params for runQL for all the cases

    t_start_RUN_QL = time.time()
    
    for eps_0 in esp0_list:
        for ALPHA in ALPHA_list:
            for dt_size in Training_traj_size_list:

                t_start_case = time.time()
                dir_path = exp + '/dt_size_' + str(dt_size) + '/ALPHA_' + str(ALPHA) + '/eps_0_' + str(eps_0) +'/'
                case+=1
                print("*******  CASE: ", case, '/', total_cases, '*******')
                print("with_guidance= ", with_guidance)
                print('eps_0 = ', eps_0)
                print('ALPHA =', ALPHA)
                print('dt_size = ', dt_size)

                # Reset Variables and environment
                # (Re)initialise Q and N based on with_guidance paramter
                # HCparams
                if with_guidance==True:
                    Q, N = initialise_guided_Q_N(g, init_Q, init_Q/2,  1) #(g, init_Qval, guiding_Qval,  init_Nval)
                else:
                    Q, N = initialise_Q_N(g,init_Q, 1) #(g, init_Qval, init_Nval)

                g.set_state(g.start_state)
                print("Q and N intialised!")


                #Learn Policy From Trajectory Data
                #if trajectory data is given, learn from it. otherwise just initilise a policy and go to refinemnet step. The latter becomes model-free QL
                if dt_size != 0:
                    Q, policy, max_delQ_list_1 = Learn_policy_from_data(paths, g, Q, N, Vx_rzns, Vy_rzns, num_of_paths=dt_size, num_actions =num_actions, ALPHA=ALPHA, method = method, num_passes = num_passes)
                    print("Learned Policy from data")

                    #Save policy
                    Policy_path = dir_path + 'Policy_01'
                    writePolicytoFile(policy, Policy_path)
                    print("Policy written to file")

                    # plot_max_Qvalues(Q, policy, X, Y, fpath = dir_path, fname = 'max_Qvalues', showfig = True)
                    print("Plotted max Qvals")

                    #Plot Policy
                    # Fig_policy_path = dir_path+'Fig_'+ 'Policy01'+'.png'
                    label_data = [F, ALPHA, init_Q, QL_Iters]
                    QL_params = policy, Q, init_Q, label_data, dir_path
                    plot_learned_policy(g, QL_params = QL_params)
                    # plot_all_policies(g, Q, policy, init_Q, label_data, full_file_path= Fig_policy_path )
                    
                    #plot max_delQ
                    plot_max_delQs(max_delQ_list_1, filename=dir_path + 'delQplot1')
                    print("plotted learned policy and max_delQs")


                else:
                    if with_guidance == True:
                        policy = initialise_policy_from_initQ(Q)
                    else:
                        policy = initialise_policy(g)


                # Times and Trajectories based on data and/or guidance
                t_list1, G0_list1, bad_count1 = plot_exact_trajectory_set(g, policy, X, Y, Vx_rzns, Vy_rzns,
                                                                       fpath = dir_path, fname = 'Trajectories_before_exp')
                print("plotted exacte trajectory set")

                # Policy Refinement Step: Learn from Experience
                Q, policy, max_delQ_list_2 = Q_learning_Iters(Q, N, g, policy, Vx_rzns, Vy_rzns, alpha=ALPHA, QIters=QL_Iters,
                                              eps_0=eps_0, eps_dec_method = eps_dec_method)
                print("Policy refined")
            
                #save Updated Policy
                Policy_path = dir_path + 'Policy_02'
                # Fig_policy_path = dir_path + 'Fig_' + 'Policy02' + '.png'

                writePolicytoFile(policy, Policy_path)
                # plot_learned_policy(g, Q, policy, init_Q, label_data, Iters_after_update=QL_Iters, full_file_path= Fig_policy_path )
                print("Refined policy written to file")

                #plots after Experince
                plot_max_delQs(max_delQ_list_2, filename= dir_path + 'delQplot2' )
                t_list2, G0_list2, bad_count2 = plot_exact_trajectory_set(g, policy, X, Y, Vx_rzns, Vy_rzns,
                                                            fpath = dir_path, fname =  'Trajectories_after_exp')
                print("plotted max delQs and exact traj set AFTER REFINEMENT")


                #Results to be printed
                # avg_time1 = np.mean(t_list1)
                # std_time1 = np.std(t_list1)
                # avg_G01 = np.mean(G0_list1)
                # avg_time2 = np.mean(t_list2)
                # std_time2 = np.std(t_list2)
                # avg_G02 = np.mean(G0_list2)
                avg_time1, std_time1, _, _ = calc_mean_and_std(t_list1)
                avg_G01, _, _, _ = calc_mean_and_std(G0_list1)
                avg_time2, std_time2, _, _ = calc_mean_and_std(t_list2)
                avg_G02, _, _, _ = calc_mean_and_std(G0_list2)

                if QL_Iters!=0:
                    bad_count1 = (bad_count1, str(bad_count1*100/dt_size)+'%')
                    bad_count2 = (bad_count2, str(bad_count2*100/dt_size) + '%')

                t_end_case = time.time()
                case_runtime = round( (t_end_case - t_start_case) / 60, 2 ) #mins

                #Print results to file
                str_Results1 = ['avg_time1','std_time1', 'bad_count1', 'avg_G01']
                Results1 = [avg_time1, std_time1, bad_count1, avg_G01]
                str_Results2 = ['avg_time2','std_time2', 'bad_count2', 'avg_G02']
                Results2 = [avg_time2, std_time2, bad_count2, avg_G02]

                Result_filename = dir_path + 'Results.txt'
                outputfile = open(Result_filename, 'w+')
                print("Before Experince ", file=outputfile)
                for i in range(len(Results1)):
                    print(str_Results1[i] + ':  ', Results1[i], file=outputfile)

                print(end="\n" * 3, file=outputfile)
                print("After Experince ", file=outputfile)
                for i in range(len(Results2)):
                    print(str_Results2[i] + ':  ', Results2[i], file=outputfile)

                print(end="\n" * 3, file= outputfile)
                print("Parameters: ", file = outputfile)
                for i in range(len(Params)):
                    print(str_Params[i] + ':  ', Params[i], file=outputfile)
                outputfile.close()

                #Print out times to file
                TrajTimes_filename = dir_path + 'TrajTimes1.txt'
                outputfile = open(TrajTimes_filename, 'w+')
                print(t_list1, file=outputfile)
                outputfile.close()

                Returns_filename = dir_path + 'G0list1.txt'
                outputfile = open(Returns_filename, 'w+')
                print(G0_list1, file=outputfile)
                outputfile.close()

                TrajTimes_filename = dir_path + 'TrajTimes2.txt'
                outputfile = open(TrajTimes_filename, 'w+')
                print(t_list2, file=outputfile)
                outputfile.close()

                Returns_filename = dir_path + 'G0list2.txt'
                outputfile = open(Returns_filename, 'w+')
                print(G0_list2, file=outputfile)
                outputfile.close()

                output_paramaters_ith_case = [exp_num, method, num_actions, nt, dt, F, startpos, endpos, eps_0, ALPHA, eps_dec_method, N_inc, dt_size, with_guidance, init_Q, num_passes, QL_Iters,
                                                avg_time1, std_time1, avg_G01, bad_count1, avg_time2, std_time2, avg_G02, bad_count2, case_runtime ]
                # Exp No	Method	Num_actions	nt	dt	F	start_pos	end_pos	Eps_0	ALPHA	dt_size_(train_size)	V[start_pos]	Mean_Time_over_5k	Variance_Over_5K	Bad Count	DP_comput_time	Mean_Glist
                # useless line now since append summary is done here itself
                output_parameters_all_cases.append(output_paramaters_ith_case) 

                append_summary_to_summaryFile( join(ROOT_DIR, 'Experiments/Exp_summary_QL.csv'),  output_paramaters_ith_case)

                RUN_QL_elpased_time = round((time.time() - t_start_RUN_QL)/60, 2)
                #Terminal Print
                print('Case_runtime= ', case_runtime)
                print('RUN_QL_elpased_time= ', RUN_QL_elpased_time, ' mins', end="\n" * 3)

    t_end_RUN_QL = time.time()
    RUN_QL_runtime = round((t_end_RUN_QL - t_start_RUN_QL)/60, 2)
    print("RUN_QL_runtime: ", RUN_QL_runtime, " mins")

    return output_parameters_all_cases


            