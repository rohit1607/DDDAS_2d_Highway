from utils.setup_grid import setup_grid
from definition import ROOT_DIR
from os.path import join
from utils.custom_functions import read_pickled_File, max_dict, picklePolicy, calc_mean_and_std, writePolicytoFile
from QL.Build_Q_from_Trajs import Q_update
import matplotlib.pyplot as plt
import copy

def sas_match(query_s1_a_s2, s1_a_r_s2):
    """
    returns true if the inputs match.
    """
    q_s1, q_a, q_s2 = query_s1_a_s2
    s1, a, r, s2 = s1_a_r_s2
    if s1 == q_s1 and s2 == q_s2 and a == q_a:
        return True
    else:
        return False


def get_similar_rzn_ids(query_s1_a_s2, sars_traj_list):
    """
    sars_traj_list = [ [(s1_a_s2), (s2_a_s3)..()] , [] , ..  [] ]
    matched_rzn_id_info = [ (traj_id, transition_id), () .... ()]
    """
    matched_rzn_id_info = []
    for sars_traj_IDX in range(len(sars_traj_list)):
        sars_traj = sars_traj_list[sars_traj_IDX]
        if sars_traj != None:
            for sars_IDX in range(len(sars_traj)):
                s1_a_r_s2 = sars_traj[sars_IDX]
                if sas_match(query_s1_a_s2, s1_a_r_s2):
                    # train_id_list[sars_traj_IDX] is the rzn id of the 5k rzns -may not use
                    # sars_traj_IDX is the trajectory index.
                    # sars_IDX is the index of a specific sars_traj where the match happened
                    # so that we dont have to search again while updating Q in future rollout
                    matched_rzn_id_info.append((sars_traj_IDX, sars_IDX))
                    break

    return matched_rzn_id_info


def update_Q_in_future_kth_rzn(g, Q, N, rzn_id_info, sars_traj_list):
    """
    rzn_id_info = (sars_traj_IDX -> idx of sars_traj_list , sars_idx_of_match)
    """
    sars_traj_IDX, sars_match_IDX = rzn_id_info
    sars_traj = sars_traj_list[sars_traj_IDX]
    max_delQ = 0
    for k in range(sars_match_IDX, len(sars_traj)):
        sars = sars_traj[k]
        Q, N, max_delQ = Q_update(Q, N, max_delQ, sars, ALPHA/10, g, N_inc)

    return Q, N


def update_Q_in_future_rollouts(g, Q, N, s1_a_s2, sars_traj_list):
    s1, a, s2 = s1_a_s2
    
    # find out which state trajectories from traindata (and hence corresponding realisations)
    # hava teh same transition of s1 -> a -> s2
    matched_rzn_id_info = get_similar_rzn_ids(s1_a_s2, sars_traj_list)
    # print("len(matched_rzn_id_info)", len(matched_rzn_id_info))
    print("matched_rzn_id_info: at transition: ", s1_a_s2, '\n', matched_rzn_id_info)
    if len(matched_rzn_id_info) != 0:
        for rzn_id_info in matched_rzn_id_info:
            Q, N = update_Q_in_future_kth_rzn(g, Q, N, rzn_id_info, sars_traj_list)
        
    return Q, N


def run_and_plot_onboard_routing_episodes(setup_grid_params, Q, N, sars_traj_list, test_id_list, fpath, fname):

    g, xs, ys, X, Y, Vx_rzns, Vy_rzns, _, _, _, _ = setup_grid_params
    # Copy Q to Qcopy

    msize = 15
    # fsize = 3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)

    minor_ticks = [i for i in range(101) if i%20!=0]
    major_ticks = [i for i in range(0,120,20)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(b= True, which='both', color='#CCCCCC', axis='both',linestyle = '-', alpha = 0.5)
    ax.tick_params(axis='both', which='both', labelsize=6)

    ax.set_xlabel('X (Non-Dim)')
    ax.set_ylabel('Y (Non-Dim)')

    st_point= g.start_state
    plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.quiver(X, Y, Vx_rzns[0, :, :], Vy_rzns[0, :, :])

    
    t_list=[]
    traj_list = []
    bad_count = 0
    # for k in range(len(test_id_list)):
    for k in range(1):
        Qcopy = copy.deepcopy(Q)
        Ncopy = copy.deepcopy(N)
        rzn = test_id_list[k]
        print("-------- In rzn ", rzn, " of test_id_list ---------")
        g.set_state(g.start_state)
        dont_plot =False
        bad_flag = False

        xtr = []
        ytr = []

        s1 = g.start_state
        t, i, j = s1
        a, q_s_a = max_dict(Qcopy[s1])

        xtr.append(g.x)
        ytr.append(g.y)
        # loop_count = 0
        # while not g.is_terminal() and g.if_within_actionable_time() and g.current_state:
        # print("__CHECK__ t, i, j")
        while True:
            # loop_count += 1
            r = g.move_exact(a, Vx_rzns[rzn, i, j], Vy_rzns[rzn, i, j])
            s2 = g.current_state()
            (t, i, j) = s2

            xtr.append(g.x)
            ytr.append(g.y)

            if g.if_edge_state((i,j)):
                bad_count += 1
                # dont_plot=True
                break
            if (not g.is_terminal()) and  g.if_within_actionable_time():
                Qcopy, Ncopy = update_Q_in_future_rollouts(g, Qcopy, Ncopy, (s1, a, s2), sars_traj_list)
                s1 = s2 #for next iteration of loop
                a, q_s_a = max_dict(Qcopy[s1])
            elif g.is_terminal():
                break
            else:
            #  i.e. not terminal and not in actinable time.
            # already checked if ternminal or not. If not terminal 
            # if time reaches nt ie not within actionable time, then increment badcount and Dont plot
                bad_count+=1
                bad_flag=True
                # dont_plot=True
                break



        if dont_plot==False:
            plt.plot(xtr, ytr)
        # if bad flag is True then append None to the list. These nones are counted later
        if bad_flag == False:  
            traj_list.append((xtr,ytr))
            t_list.append(t)
        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list.append(None)


    if fname != None:
        plt.savefig(join(fpath,fname),bbox_inches = "tight", dpi=200)
        plt.cla()
        plt.close(fig)
        picklePolicy(traj_list, join(fpath,fname))
        print("*** pickled phase2 traj_list ***")

    return t_list, bad_count



def run_onboard_routing_for_test_data(exp_num_case_dir, setup_grid_params):
   
    global ALPHA
    global N_inc
    Q = read_pickled_File(join(exp_num_case_dir, 'Q2'))
    N = read_pickled_File(join(exp_num_case_dir, 'N2'))
    test_id_list = read_pickled_File(join(exp_num_case_dir, 'test_id_list'))
    train_id_list = read_pickled_File(join(exp_num_case_dir, 'train_id_list'))
    sars_traj_list = read_pickled_File(join(exp_num_case_dir, 'sars_traj_Trained_Trajectories_after_exp'))
    train_output_params = read_pickled_File(join(exp_num_case_dir, 'output_paramaters'))

    ALPHA = train_output_params[9]
    N_inc = train_output_params[11]
    print("ALPHA, N_inc = ", ALPHA, N_inc)
    print('len(sars_traj_list) = ', len(sars_traj_list))
    t_list, bad_count = run_and_plot_onboard_routing_episodes(setup_grid_params, Q, N,
                                             sars_traj_list, test_id_list, exp_num_case_dir, 'Phase2_trajs_QNcopy_break' )

    phase2_results = calc_mean_and_std(t_list)
    picklePolicy(phase2_results,join(exp_num_case_dir, 'Phase2_trajs_QNcopy_break'))
    writePolicytoFile(phase2_results,join(exp_num_case_dir, 'Phase2_trajs_QNcopy_break'))
    avg_time_ph2, std_time_ph2, cnt_ph2 , none_cnt_ph2 = phase2_results
    print("avg_time_ph2", avg_time_ph2,'\n', 
           "std_time_ph2", std_time_ph2, '\n',
            "cnt_ph2",cnt_ph2 , '\n',
           "none_cnt_ph2", none_cnt_ph2)

    
    return



setup_grid_params = setup_grid(num_actions=16, nt = 100)
rel_path = 'Experiments/72/QL/num_passes_50/QL_Iter_x1/dt_size_4800/ALPHA_0.05/eps_0_0.33'
exp_num_case_dir = join(ROOT_DIR, rel_path)

run_onboard_routing_for_test_data(exp_num_case_dir, setup_grid_params)