from utils.setup_grid import setup_grid
from utils.custom_functions import createFolder, append_summary_to_summaryFile, read_pickled_File
from os import getcwd, makedirs
from os.path import join, exists
import pickle
import time

#initialise dictionary for storing counts.
#transition_dict is a nested dictionary
def initialise_dict(g):
    transition_dict = {}
    for s in state_list:
        transition_dict[s]={}
        for a in g.actions:
            transition_dict[s][a]={}

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

def get_S_from_S_id(S_id, gsize):
    """
    returns (i,j) from combined state ID. this is for 2d
    :param S_id: combined id of state- one number encapsulating t,i,j  or i,j
    :param gsize: dim of grid in 1direction
    :return: S in t
    TODO: Add t for 3d case
    """

    i = S_id // gsize
    j = S_id % gsize
    S = (i,j)

    return S

def convert_COO_to_dict(tdict, g, coo_file, Rsa_file):
    """
    takes in saved coo and Rsa files. unpacks them and converts them to dict
    :param init_transition_dict:
    :param g:
    :param coo_file:
    :param Rsa_file:
    :return:
    TODO: add another loop for T for 3d case
    """
    coo_list = read_pickled_File(coo_file)
    Rs_list = read_pickled_File(Rsa_file)
    num_actions = len(g.actions)

    for i in range(num_actions):
        coo = coo_list[i]
        Rs = Rs_list[i]
        m, n = coo.shape
        a = g.actions[i]
        for k in range(n):
            S1 = get_S_from_S_id(coo[0, k], g.ni)
            S2 = get_S_from_S_id(coo[1, k], g.ni)
            prob = coo[2,k]
            r = Rs[int(coo[0,k])]
            try:
                tdict[S1][a][S2] = (prob, r)
            except:
                print(S1, 'is not an actionable state')

    return tdict



def get_Model_from_COO(coo_filename, rs_filename, filename = 'Transition_dict',  n_actions = 1, nt = None, dt =None, F =None, startpos = None, endpos = None):

    print("Building Model")
    global state_list
    global base_path
    global save_path

    start_time = time.time()

    #setup grid
    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, params, param_str = setup_grid(num_actions=n_actions)

    #name of pickle file containing transtion prob in dictionary format
    filename =  filename + str(n_actions) + 'a'
    base_path = join(getcwd(),'DP/Trans_matxs/')
    save_path = base_path + filename
    if exists(save_path):
        print("Folder Already Exists !!")
        return

    #build probability transition dictionary
    state_list = g.ac_state_space()
    init_transition_dict = initialise_dict(g)
    transition_dict = convert_COO_to_dict(init_transition_dict, g, coo_filename, rs_filename)
    build_time = time.time() - start_time

    #save dictionary to file
    data = params, param_str, g.reward_structure, build_time
    write_files(transition_dict, filename, data)
    total_time = time.time() - start_time

    #command line outputs
    print("Dictionary saved !")
    print("Build Time = ", build_time/60, " mins")
    print("Total TIme = ", total_time/60, "mins")

coo_file = '/Users/rohit/workspace/ROHIT/pycuda_transition_clac_tests/H2Df_coo'
rs_file = '/Users/rohit/workspace/ROHIT/pycuda_transition_clac_tests/H2Df_rsa'

get_Model_from_COO( coo_file, rs_file, filename='fromCOO_a', n_actions=16)
# Build_Model(filename='Model_1_',n_actions=8)
# Build_Model(filename='Model_1_',n_actions=16)
