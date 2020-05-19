from utils.setup_grid import setup_grid
from utils.custom_functions import createFolder, append_summary_to_summaryFile
from os.path import join, exists
import pickle
import time
from definition import ROOT_DIR
import numpy as np

#initialise dictionary for storing counts.
#transition_dict is a nested dictionary
def initialise_dict(g):
    transition_dict = {}
    for s in state_list:
        transition_dict[s]={}
        for a in g.actions:
            transition_dict[s][a]={}

    return transition_dict


#populate transition_dict with counts
def compute_transition_probability_and_rewards(transition_dict, g, num_rzns, Vx_rzns, Vy_rzns):
    s_count = 0
    pr_intvl = 100
    a_time = time.time()
    for s in state_list:
        n_states = len(state_list)
        s_count += 1
        t0, i0, j0 = s
        if s_count % pr_intvl == 0:
            b_time = (time.time() - a_time)/60 # time taken per print interval in mins
            t_left = np.round(b_time * (n_states - s_count)/ pr_intvl , 3 ) #expected time left in minutes
            a_time = time.time()
            print("s_count: ", s_count, " /  ", n_states, "  ----  ", np.round(b_time, 3), "min  ----  ", t_left, " mins left" )
        for a in g.actions:
            for rzn in range(num_rzns):
                g.set_state(s)
                r = g.move_exact(a, Vx_rzns[rzn, i0, j0], Vy_rzns[rzn, i0, j0])
                s_new = g.current_state()
                if transition_dict[s][a].get(s_new):
                    transition_dict[s][a][s_new][0] += 1
                    transition_dict[s][a][s_new][1] += (1/transition_dict[s][a][s_new][0])*(r - transition_dict[s][a][s_new][1])
                else:
                    transition_dict[s][a][s_new] = [1, r]

    #convert counts to probabilites
    for s in state_list:
        for a in g.actions:
            for s_new in transition_dict[s][a]:
                transition_dict[s][a][s_new][0] = transition_dict[s][a][s_new][0]/num_rzns

    return transition_dict


def write_files(transition_dict, filename, data):
    """
    Pickles dictionary contaniing model details.
    Writes parameters to file.
    Writes parameters to summary file
    :param transition_dict:
    :param filename:
    :return:
    """

    summary_file = base_path + 'model_summary.csv'
    params, param_str, reward_structure, build_time = data

    createFolder(save_path)

    # save transition_probs. Pickle it.
    with open(save_path + '/' + filename + '.p', 'wb') as fp:
        pickle.dump(transition_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_path + '/' + filename + '_params.txt', 'w') as f:
        for i in range(len(param_str)):
            f.write(param_str[i] + ':' + '    ' + str(params[i]) + "\n")
        f.write("Reward Structure: " + str(reward_structure) + "\n")
        f.write("Build Time: "+ str(build_time))

    # append_summary_to_summaryFile(summary_file, )




def Build_Model(filename = 'Transition_dict', n_actions = 1, nt = None, dt =None, F =None, startpos = None, endpos = None, Test_grid =False):

    print("Building Model")
    global state_list
    global base_path
    global save_path

    start_time = time.time()

    #setup grid
    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, params, param_str = setup_grid(num_actions=n_actions, Test_grid= Test_grid)

    #name of pickle file containing transtion prob in dictionary format
    filename =  filename + str(n_actions) + 'a'
    base_path = join(ROOT_DIR,'DP/Trans_matxs_3D/')
    save_path = base_path + filename
    if exists(save_path):
        print("Folder Already Exists !!")
        return

    #build probability transition dictionary
    state_list = g.ac_state_space()
    init_transition_dict = initialise_dict(g)
    transition_dict = compute_transition_probability_and_rewards(init_transition_dict, g, num_rzns, Vx_rzns, Vy_rzns)
    build_time = time.time() - start_time

    #save dictionary to file
    data = params, param_str, g.reward_structure, build_time
    write_files(transition_dict, filename, data)
    total_time = time.time() - start_time

    #command line outputs
    print("Dictionary saved !")
    print("Build Time = ", build_time/60, " mins")
    print("Total TIme = ", total_time/60, "mins")


# Build_Model(filename='TestModel_1_',n_actions=8, Test_grid=True)
# Build_Model(filename='Model_1_',n_actions=8)
# Build_Model(filename='Model_1_',n_actions=16)
