import time
import numpy as np
from utils.custom_functions import picklePolicy
import pickle
from utils.plot_functions import plot_exact_trajectory, plot_exact_trajectory_set, plot_learned_policy
from scipy import sparse

action_state_space = []
"""
__________________________________FUNCTIONS_______________________________________________________--
"""
# initialise value functions and policy
def initialise_policy_and_V(g):
    policy = {}
    V = {}
    for s in g.state_space():
        V[s] = 0
        policy[s] =(0,0)

    return policy, V

def value_iteration_update(g, V, Trans_prob):
    delV = -10
    state_flag = None
    for s in action_state_space:
        old_V = V[s]
        g.set_state(s)
        best_val = -float('inf')
        for a in g.actions:
            val = 0
            for s_new in Trans_prob[s][a]:
                prob, r = Trans_prob[s][a][s_new]
                val += prob * (r + V[s_new])
                # prob, _ = Trans_prob[s][a][s_new]
                # val += prob * (g.R(s, s_new) + V[s_new])
            if val > best_val:
                best_val = val
        V[s] = best_val
        #         print((delV, np.abs(V[s]-old_V)))
        if delV <= np.abs(V[s]-old_V):
            state_flag = s
            delV = np.abs(V[s]-old_V)

        # delV = max(delV, np.abs(V[s] - old_V))

    return V, delV, state_flag


def VI_update_SpMV_format(g, V, sp_TPM_alist, R_s_alist):
    """
    Performs VI update using Sparse Matrix Vector (SpMV) multiplication
    :param sp_TPM_alist: list of sparse TPM for each action
    :param R_s_alist: list of Reward(s) vector for each each action
    :return: converged Value function, V,
    TODO: check edge state and terminal state values and see if they interfere with logic
    """
    # Initialise Value Fn vector of size = no. of states = g.ni*g.nj for 2d problem
    num_actions = len(g.actions)
    V_a_mat = np.zeros( ((g.ni * g.nj),num_actions) )

    old_V = V
    # compute 1step update for each action and store V vector of each action in a column of V_a_mat
    for i in range(num_actions):
        V_a_mat[:,i] = R_s_alist[i] + sp_TPM_alist[i].dot(V[:,i])

    # for each state (element of V), find max of each row, i.e max V(s) across all actions
    for i in range(len(V)):
        V[i] = np.max(V[i,:])

    delV_vector = np.abs(V - old_V)
    state_flag = np.argmax(delV_vector)
    delV = delV_vector[state_flag]

    return V, delV, state_flag




def compute_Policy(g, policy, Trans_prob, V):
    for s in action_state_space:
        g.set_state(s)
        best_val = -float('inf')
        new_a = None
        for a in g.actions:
            val = 0
            for s_new in Trans_prob[s][a]:
                prob, r = Trans_prob[s][a][s_new]
                val += prob * (r + V[s_new])
                # prob, _ = Trans_prob[s][a][s_new]
                # val += prob * (g.R(s, s_new) + V[s_new])
            if val > best_val:
                best_val = val
                new_a = a
        policy[s] = new_a

    return policy


def pickle_V(V):
    with open('ValueFunc.p', 'wb') as fp:
        pickle.dump(V, fp, protocol=pickle.HIGHEST_PROTOCOL)

def write_list_to_file(list, file):
    outputfile = open(file, 'w+')
    print(list, file=outputfile)
    outputfile.close()



"""
__________________________________MAIN_______________________________________________________--
"""

def run_DP(setup_grid_params, prob_file, output_file, output_path, threshold = 1e-3, eg_rzn =1):

    #Set up grid
    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, params, param_str = setup_grid_params

    global action_state_space
    action_state_space = g.ac_state_space()

    #Read transition probability
    with open('DP/Trans_matxs/' + prob_file + '/' + prob_file + '.p', 'rb') as fp:
        Trans_prob = pickle.load(fp)

    #Initialise Policy and V
    policy, V = initialise_policy_and_V(g)
    countb = 0

    start = time.time()
    #Iterate VI updates
    while True:
        countb += 1

        V, del_V_max, flagged_state = value_iteration_update(g, V, Trans_prob)

        if countb % 100 == 0:
            print("iter: ", countb, del_V_max, flagged_state)

        if del_V_max< threshold:
            print("Converged after ", countb, " iterations")
            break

    # Compute policy
    policy = compute_Policy(g, policy, Trans_prob, V)
    end = time.time()
    DP_compute_time = end - start
    # Save policy to file
    picklePolicy(policy, output_path + '/policy')

    trajectory, G = plot_exact_trajectory(g, policy, X, Y, Vx_rzns[eg_rzn,:,:], Vy_rzns[eg_rzn,:,:], output_path, fname='Sample_Traj_with_policy_in rzn_'+ str(eg_rzn), lastfig = True)
    tlist, Glist, badcount = plot_exact_trajectory_set(g, policy, X, Y, Vx_rzns, Vy_rzns, output_path, fname='Traj_set' + output_file)

    plot_learned_policy(g, DP_params = [policy, output_path])

    write_list_to_file(tlist, output_path+'/tlist')
    write_list_to_file(Glist, output_path +'/Glist')

    return V[g.startpos], np.mean(tlist), np.std(tlist), badcount, DP_compute_time, np.mean(Glist)