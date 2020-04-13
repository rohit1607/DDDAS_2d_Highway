from utils.setup_grid import setup_grid
from utils.custom_functions import createFolder, append_summary_to_summaryFile
from os.path import join, exists
import pickle
import time
from definition import ROOT_DIR

#initialise dictionary for storing counts.
#transition_dict is a nested dictionary
def initialise_dict(g):
    transition_dict = {}
    for s in state_list:
        transition_dict[s]={}
        for a in g.actions:
            transition_dict[s][a]={}

    return transition_dict


def compute_transition_probability_and_rewards(transition_dict, g, num_rzns, Vx_rzns, Vy_rzns, method='const_rzn'):
    """
    IMPORTANT: There are two methods of constructing the transition function.
    :param transition_dict:
    :param g:
    :param num_rzns:
    :param Vx_rzns:
    :param Vy_rzns:
    :param method: 1. 'const_rzn' - if the agent is in the velocity field of 1 rzn, then after doing an action,
                        it remains in the same realisation.
                   2. 'var_rzn' - if the agent is in the velocity field of 1 rzn, then after doing an action,
                        it has probability of experiencing velocity field of another realisation.
    :return:
    """

    if method == 'const_rzn':
        for rzn in range(num_rzns):
            print("rzn: ", rzn)
            for pos in pos_list:
                i, j = pos
                vx = Vx_rzns[rzn, i, j]
                vy = Vy_rzns[rzn, i, j]
                dvx = g.discretize(vx, g.vxs)
                dvy = g.discretize(vy, g.vys)
                s = (i, j, dvx, dvy)
                # print()
                # print("----- s = ", s, "----")
                for a in g.actions:
                    # print("action ", a)
                    g.set_state(s)
                    r = g.move_exact(a)
                    i2, j2 = g.current_pos()
                    if g.if_edge_state((i2,j2)):#agent may end up in edge state where the velocity it reads mya be NaN
                        vx2 = 0
                        vy2 = 0
                    else:
                        vx2 = Vx_rzns[rzn, i2, j2]
                        vy2 = Vy_rzns[rzn, i2, j2]
                    # print("test in compute_trans:i2, j2, vx2, vy2 ", i2, j2, vx2, vy2)
                    dvx2 = g.discretize(vx2, g.vxs)
                    dvy2 = g.discretize(vy2, g.vys)
                    s_new = (i2, j2, dvx2, dvy2)
                    if transition_dict[s][a].get(s_new):
                        transition_dict[s][a][s_new][0] += 1
                        transition_dict[s][a][s_new][1] += (1 / transition_dict[s][a][s_new][0]) * (r - transition_dict[s][a][s_new][1])
                    else:
                        transition_dict[s][a][s_new] = [1, r]

        # convert counts to probabilites
        for s in state_list:
            for a in g.actions:
                for s_new in transition_dict[s][a]:
                    transition_dict[s][a][s_new][0] = transition_dict[s][a][s_new][0] / num_rzns


    elif method == 'var_rzn':

        pass

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
    global pos_list

    start_time = time.time()

    #setup grid
    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, params, param_str = setup_grid(num_actions=n_actions, Test_grid= Test_grid)

    #name of pickle file containing transtion prob in dictionary format
    filename =  filename + str(n_actions) + 'a'
    base_path = join(ROOT_DIR,'DP/Trans_matxs/')
    save_path = base_path + filename
    if exists(save_path):
        print("Folder Already Exists !!")
        return

    #build probability transition dictionary
    state_list = g.ac_state_space()
    pos_list = g.ac_state_space(only_pos=True)
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


Build_Model(filename='4st_Model_1', n_actions=16)
# Build_Model(filename='Model_1_',n_actions=8)
# Build_Model(filename='Model_1_',n_actions=16)
