from QL.Build_Q_from_Trajs import Learn_policy_from_data
from utils.plot_functions import plot_max_delQs, plot_exact_trajectory_set, plot_max_Qvalues, plot_learned_policy
from QL.Q_Learning import Q_learning_Iters
from utils.custom_functions import initialise_policy, initialise_Q_N, initialise_guided_Q_N, writePolicytoFile, initialise_policy_from_initQ, createFolder
import time
import math
import numpy as np
from definition import ROOT_DIR
from os.path import join
Pi = math.pi


def run_QL(setup_grid_params, QL_params, QL_path):
    
    exp =  QL_path

#Parameters
# Training_traj_size_list = [1000, 2000, 3000, 4000, 5000]
# ALPHA_list = [0.2, 0.35, 0.5, 0.75]
# esp0_list = [0.25, 0.5, 0.75]

# with_guidance = True
# Training_traj_size_list = [5000]
# ALPHA_list = [0.5]
# esp0_list = [0.5]
# QL_Iters = int(1000)
# 
# init_Q = -1000000
#     stream_speed = 0.2

    Training_traj_size_list, ALPHA_list, esp0_list, QL_Iters, init_Q, with_guidance, method, num_passes = QL_params
    
    #Read data from files
    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, paths, params, param_str = setup_grid_params
    print("In TQLearn: ", len(params), params)
    num_actions, nt, dt, F, startpos, endpos = params
    
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

    case =0
    start =time.time()
    for eps_0 in esp0_list:
        for ALPHA in ALPHA_list:
            for dt_size in Training_traj_size_list:

                dir_path = exp + '/dt_size_' + str(dt_size) + '/ALPHA_' + str(ALPHA) + '/eps_0_' + str(eps_0) +'/'
                case+=1
                print("*******  CASE: ", case, '/', total_cases, '*******')
                print("with_guidance= ", with_guidance)
                print('eps_0 = ', eps_0)
                print('ALPHA =', ALPHA)
                print('dt_size = ', dt_size)


                #Reset Variables and environment
                if with_guidance==True:
                    Q, N = initialise_guided_Q_N(g, init_Q, init_Q/2,  1)
                else:
                    Q, N = initialise_Q_N(g,init_Q, 1)
                g.set_state(g.start_state)


                #Learn Policy From Trajectory Data
                if dt_size != 0:
                    Q, policy, max_delQ_list_1 = Learn_policy_from_data(paths, g, Q, N, Vx_rzns, Vy_rzns, num_of_paths=dt_size, num_actions =num_actions, ALPHA=ALPHA, method = method, num_passes = num_passes)
                    plot_max_Qvalues(Q, policy, X, Y)
                    #Save policy
                    Policy_path = dir_path + 'Policy_01'

                    #Plot Policy
                    # Fig_policy_path = dir_path+'Fig_'+ 'Policy01'+'.png'
                    label_data = [F, ALPHA, init_Q, QL_Iters]
                    QL_params = policy, Q, init_Q, label_data, dir_path
                    plot_learned_policy(g, QL_params = QL_params)
                    # plot_all_policies(g, Q, policy, init_Q, label_data, full_file_path= Fig_policy_path )

                    writePolicytoFile(policy, Policy_path)
                    plot_max_delQs(max_delQ_list_1, filename=dir_path + 'delQplot1')
                else:
                    if with_guidance == True:
                        policy = initialise_policy_from_initQ(Q)
                    else:
                        policy = initialise_policy(g)


                #Times and Trajectories based on data and/or guidance
                t_list1, G0_list1, bad_count1 = plot_exact_trajectory_set(g, policy, X, Y, Vx_rzns, Vy_rzns, exp,
                                                                       fname=dir_path + 'Trajectories_before_exp')

                #Learn from Experience
                Q, policy, max_delQ_list_2 = Q_learning_Iters(Q, N, g, policy, Vx_rzns, Vy_rzns, alpha=ALPHA, QIters=QL_Iters,
                                              eps_0=eps_0)

                #save Updated Policy
                Policy_path = dir_path + 'Policy_02'
                # Fig_policy_path = dir_path + 'Fig_' + 'Policy02' + '.png'

                writePolicytoFile(policy, Policy_path)
                # plot_learned_policy(g, Q, policy, init_Q, label_data, Iters_after_update=QL_Iters, full_file_path= Fig_policy_path )

                #plots after Experince
                plot_max_delQs(max_delQ_list_2, filename= dir_path + 'delQplot2' )
                t_list2, G0_list2, bad_count2 = plot_exact_trajectory_set(g, policy, X, Y, Vx_rzns, Vy_rzns, exp,
                                                            fname=dir_path + 'Trajectories_after_exp')

                #Results to be printed
                avg_time1 = np.mean(t_list1)
                std_time1 = np.std(t_list1)
                avg_G01 = np.mean(G0_list1)
                avg_time2 = np.mean(t_list2)
                std_time2 = np.std(t_list2)
                avg_G02 = np.mean(G0_list2)

                if QL_Iters!=0:
                    bad_count1 = (bad_count1, str(bad_count1*100/dt_size)+'%')
                    bad_count2 = (bad_count2, str(bad_count2*100/dt_size) + '%')

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


                end = time.time()
                time_taken = round(end -start,2)
                #Terminal Print
                print('time_taken= ', time_taken, 's', end="\n" * 3)

                end = start