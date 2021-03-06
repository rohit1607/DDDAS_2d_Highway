from grid_world_stationary import timeOpt_grid
from utils.custom_functions import *
from utils.plot_functions import plot_exact_trajectory, plot_learned_policy
from os.path import join
from os import getcwd
from definition import N_inc
# ALPHA =0.1


def Run_Q_learning_episode(g, Q, N, ALPHA, Vx, Vy, eps):
    # print()
    # print("new episode")

    s1 = g.start_state
    i,j =s1
    g.set_state(s1)
    policy=None
    a1 = stochastic_action_eps_greedy(policy, s1, g, eps, Q=Q)
    count = 0
    max_delQ = 0

    # while not g.is_terminal() and g.if_within_TD_actionable_time():
    while not g.if_edge_state((s1)) and g.if_within_actionable_time():
        """Will have to change this for general time"""
        r = g.move_exact(a1, Vx[i,j], Vy[i,j])
        s2 = g.current_state()
        # if g.is_terminal() or (not g.if_within_actionable_time()):
        alpha = ALPHA / N[s1][a1]
        N[s1][a1] += N_inc

        #maxQsa = 0 if next state is a terminal state
        max_q_s2_a2= 0
        # if (s2[1],s2[2])!=g.endpos:
        if s2 != g.endpos and not g.if_edge_state(s2):
            a2, max_q_s2_a2 = max_dict(Q[s2])

        old_qsa = Q[s1][a1]
        Q[s1][a1] = Q[s1][a1] + alpha*(r + max_q_s2_a2 - Q[s1][a1])

        if np.abs(old_qsa - Q[s1][a1]) > max_delQ:
            max_delQ = np.abs(old_qsa - Q[s1][a1])

        #action for next iteration
        # if (s2[1], s2[2]) != g.endpos:


        # if (s2[0], s2[1]) != g.endpos:
        if s2 != g.endpos and not g.if_edge_state(s2):
            a1 = stochastic_action_eps_greedy(policy, s2, g, eps, Q=Q)
        else:
            break

        s1 = s2
        i,j = s1

    return Q, N, max_delQ


def Q_learning_Iters(Q, N, g, policy, vx_rlzns, vy_rlzns, alpha = 0.5, QIters=10000, stepsize=1000, post_train_size = 1000, eps_0=0.5):

    max_delQ_list=[]
    t=1
    for k in range(QIters):
        # alpha = 1/(k+1)
        # if k%(QIters/500)==0:
        #     t+=0.04
        t+=1/QIters
        eps = eps_0/t

        if k%500==0:
            print("Qlearning Iters: iter, eps =",k, eps)

        rzn = k%5000
        Vx = vx_rlzns[rzn,:,:]
        Vy = vy_rlzns[rzn,:,:]
        Q, N, max_delQ = Run_Q_learning_episode(g, Q, N,alpha, Vx, Vy, eps)
        if k%500==0:
            max_delQ_list.append(max_delQ)

    if QIters!=0:
        for s in Q.keys():
            newa, _ = max_dict(Q[s])
            # if policy[s] != newa:
            #     print("s, old policy, new policy",s, policy[s], newa)
            policy[s]=newa

    return Q, policy, max_delQ_list


# output_path, exp_num = create_new_dir()          #dirs Exp/1, Exp/2, ...
# DP_path = join(output_path,'DP')                 #dirs Exp/1/DP
# QL_path = join(output_path,'QL')


def test_QL(QIters=10000):
    QL_path = getcwd()
    xs = np.arange(10)
    ys = xs
    dt=1
    vStream_x = np.zeros((5000, len(ys), len(xs)))
    vStream_y = np.zeros((5000, len(ys), len(xs)))
    stream_speed =0.75
    vStream_x[0,4:7,:]=stream_speed
    F=1

    X, Y = my_meshgrid(xs, ys)
    print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

    g = timeOpt_grid(xs, ys, dt, 100, F, (8,4), (1,4), num_actions=16 )

    action_states = g.ac_state_space()

    # initialise policy
    policy = initialise_policy(g)

    # initialise Q and N
    init_Q=0
    Q, N = initialise_Q_N(g, init_Q, 1)

    print("Teswt")
    Q, policy, max_delQ_list = Q_learning_Iters(Q, N, g, policy, vStream_x, vStream_y, QIters=QIters, eps_0 = 1)

    print("shapes of X, Y, vStream",X.shape, Y.shape, vStream_x.shape, vStream_x[0,:,:].shape)
    traje, return_value = plot_exact_trajectory(g, policy, X, Y, vStream_x[0,:,:], vStream_y[0,:,:], QL_path, fname='QLearning', lastfig=True)
    ALPHA=None
    label_data = [ F, stream_speed, ALPHA, init_Q, QIters ]
    plot_learned_policy(g, Q, policy, init_Q, label_data)


# test_QL()