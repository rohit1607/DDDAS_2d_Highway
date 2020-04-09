from utils.custom_functions import Calculate_action, max_dict, initialise_policy
import numpy as np
from definition import N_inc, Sampling_interval
import random
from _collections import OrderedDict

SAMPLING_Interval = Sampling_interval

# Helper func to compute cell state of pt in trajectory and add it to dic
def compute_cell(grid, s):
    remx = (s[0] - grid.xs[0]) % grid.dj
    remy = -(s[1] - grid.ys[-1]) % grid.di
    xind = (s[0] - grid.xs[0]) // grid.dj
    yind = -(s[1] - grid.ys[-1]) // grid.di

    if remx >= 0.5 * grid.dj and remy >= 0.5 * grid.di:
        xind+=1
        yind+=1
    elif remx >= 0.5 * grid.dj and remy < 0.5 * grid.di:
        xind+=1
    elif remx < 0.5 * grid.dj and remy >= 0.5 * grid.di:
        yind+=1

    return int(yind), int(xind)


#Checks if pts p1 and p2 lie within the same cell
def within_same_spatial_cell(grid , point1, point2):
    state1 = compute_cell(grid, point1)
    state2 = compute_cell(grid, point2)
    if(state1 == state2) :
        return 1
    else:
        return 0

def build_experience_buffer(grid, Vx_rzns, Vy_rzns, paths, sampling_interval, num_of_paths, num_actions ):
    exp_buffer_all_trajs = []
    for k in range(num_of_paths):
        exp_buffer_kth_traj = []
        Vxt = Vx_rzns[k, :, :]
        Vyt = Vy_rzns[k, :, :]
        trajectory = paths[0, k]

        # append starting point to traj
        coord_traj = [(trajectory[0][0], trajectory[0][1])]
        s_i, s_j = compute_cell(grid, trajectory[0])
        state_traj = [(s_i, s_j)]


        # make dictionary states mapping to coords. and choose middle coord to append to traj
        traj_dict = OrderedDict()
        for j in range(0, len(trajectory)):
            s_i, s_j = compute_cell(grid, trajectory[j])
            s = (s_i, s_j)
            c = (trajectory[j][0], trajectory[j][1])
            if not traj_dict.get(s):
                traj_dict[s] = [c]
            else:
                traj_dict[s].append(c)
        keys = list(traj_dict.keys())
        keys.remove(keys[0])        #remove first and last keys (states).
        keys.remove(keys[-1])          #They are appended separately

        for s in keys:
            state_traj.append(s)
            l = len(traj_dict[s])
            coord_traj.append(traj_dict[s][int(l//2)])

        coord_traj.append((trajectory[-1][0], trajectory[-1][1]))
        s_i, s_j = compute_cell(grid, trajectory[-1])
        state_traj.append((s_i, s_j))

        state_traj.reverse()
        coord_traj.reverse()

        #build buffer
        print("check warning: ", k)
        # print("s1, p1, p2, Vxt, Vyt")
        for i in range(len(state_traj)-1):
            s1=state_traj[i+1]
            s2=state_traj[i]
            # t ,m,n=s1
            m, n = s1
            p1=coord_traj[i+1]
            p2=coord_traj[i]
            """COMMENTING THIS STATEMENT BELOW"""
            # if (s1[1],s1[2])!=(s2[1],s2[2]):
            #vx=Vxt[t,i,j]
            vx = Vxt[m, n]
            vy = Vyt[m, n]
            # print(s1,p1,p2, vx, vy)
            r1 = grid.move_exact(a1, vx, vy)
            exp_buffer_kth_traj.append([s1, a1, r1, s2])

        #append kth-traj-list to master list
        exp_buffer_all_trajs.append(exp_buffer_kth_traj)

    return exp_buffer_all_trajs


"""
## original . doesnt do away with copies. paths input not pruned and padded.
# Calculate theta (outgoing angle) between last point in 1st cell and first point in next cell
def build_experience_buffer(grid, Vx_rzns, Vy_rzns, paths, sampling_interval, num_of_paths, num_actions ):
    exp_buffer_all_trajs = []
    for k in range(num_of_paths):
        exp_buffer_kth_traj = []
        Vxt = Vx_rzns[k, :, :]
        Vyt = Vy_rzns[k, :, :]
        trajectory = paths[0, k]
        state_traj = []
        coord_traj = []

        #build sub sampled trajectory and reverse it
        for j in range(0, len(trajectory) - 1, sampling_interval):  # the len '-1' is to avoid reading NaN at the end of path data
            s_i, s_j = compute_cell(grid, trajectory[j])

            # state_traj.append((s_t, s_i, s_j))
            # coord_traj.append((grid.ts[s_t],trajectory[j][0], trajectory[j][1]))
            state_traj.append((s_i, s_j))
            coord_traj.append((trajectory[j][0], trajectory[j][1]))
        state_traj.reverse()
        coord_traj.reverse()

        # Append first state to the sub sampled trajectory
        m, n = grid.start_state
        x0 = grid.xs[n]
        y0 = grid.ys[grid.ni - 1 - m]
        state_traj.append(grid.start_state)
        # coord_traj.append((grid.ts[p],x0,y0))
        coord_traj.append((x0, y0))

        #build buffer
        for i in range(len(state_traj)-1):
            s1=state_traj[i+1]
            s2=state_traj[i]
            # t ,m,n=s1
            m, n = s1
            p1=coord_traj[i+1]
            p2=coord_traj[i]
       
            # if (s1[1],s1[2])!=(s2[1],s2[2]):
            #vx=Vxt[t,i,j]
            a1 = Calculate_action(s1,p1,p2, Vxt, Vyt, num_actions)
            r1 = grid.move_exact(a1, Vxt[m, n], Vyt[m, n])
            exp_buffer_kth_traj.append([s1, a1, r1, s2])

        #append kth-traj-list to master list
        exp_buffer_all_trajs.append(exp_buffer_kth_traj)

    return exp_buffer_all_trajs
"""

def learn_Q_from_exp_buffer(grid, exp_buffer, Q, N, ALPHA, method='reverse_order', num_passes =1):
    """
    Learns Q values after building experience buffer. Contains 2 types of methods- 1.reverse pass through buffer  2.random pass through buffer
    :param grid:
    :param exp_buffer:
    :param Q:
    :param N:
    :param ALPHA:
    :param method:
    :param num_passes:
    :return:
    """

    def Q_update(Q, N, max_delQ, sars):
        s1, a1, r1, s2 = sars
        if s1 != grid.endpos:
            N[s1][a1] += N_inc
            alpha1 = ALPHA / N[s1][a1]
            q_s1_a1 = r1
            if s2 != grid.endpos:
                _, val = max_dict(Q[s2])
                q_s1_a1 = r1 + val
            old_qsa = Q[s1][a1]
            Q[s1][a1] += alpha1 * (q_s1_a1 - Q[s1][a1])
            delQ = np.abs(old_qsa - Q[s1][a1])
            if delQ > max_delQ:
                max_delQ = delQ
        return Q, N, max_delQ

    if not (method == 'reverse_order' or method == 'iid'):
        print("No such method learning Q values from traj")
        return

    print("In Build_Q_...   learning method = ", method)
    max_delQ_list = []

    if method == 'reverse_order':
        for Pass in range(num_passes):
            print("in Build_Q_.. : pass ", Pass)
            max_delQ = 0
            for kth_traj_buffer in exp_buffer:
                for sars in kth_traj_buffer:
                    Q, N, max_delQ = Q_update(Q, N, max_delQ, sars)

            max_delQ_list.append(max_delQ)
            print('max_delQ= ',max_delQ)
            print("Q[start] = ", Q[grid.startpos])
            print('Q[s]: best a, val =', max_dict(Q[grid.startpos]))
            if max_delQ < 1:
                print("Qs converged")
                break

    if method == 'iid':
        flatten = lambda l: [item for sublist in l for item in sublist]
        exp_buffer = flatten(exp_buffer)
        idx_list= np.arange(len(exp_buffer))
        print(len(exp_buffer))

        for Pass in range(num_passes):
            print("in Build_Q_.. : pass ", Pass)
            random.shuffle(idx_list)
            max_delQ = 0
            for i in idx_list:
                sars = exp_buffer[i]
                Q, N, max_delQ = Q_update(Q, N, max_delQ, sars)

            max_delQ_list.append(max_delQ)
            print('max_delQ= ', max_delQ)
            print("Q[start] = ", Q[grid.startpos])
            print('Q[s]: best a, val =', max_dict(Q[grid.startpos]))
            if max_delQ < 1:
                print("Qs converged")
                break

    return Q, max_delQ_list


def learn_Q_from_trajs(paths, grid, Q, N,  Vx_rzns, Vy_rzns, num_of_paths, num_actions, ALPHA, sampling_interval, method= 'reverse_order', num_passes= 1):
    exp_buffer = build_experience_buffer(grid, Vx_rzns, Vy_rzns, paths, sampling_interval, num_of_paths, num_actions)
    Q, max_delQ_list = learn_Q_from_exp_buffer(grid, exp_buffer, Q, N, ALPHA, method=method, num_passes = num_passes)
    return Q, max_delQ_list


def EstimateQ_mids_mids2(paths, grid, Q, N,  Vx_rzns, Vy_rzns, num_of_paths, num_actions, ALPHA, sampling_inerval):
    # considers transition from middle of state to middle of state
    # chooses correct actions by taking into consideration velocity field
    # generates velocity field realization here

    max_delQ_list=[]
    #pick trajectory from paths and store in reverse order
    for k in range(num_of_paths):
        if k%500 == 0:
            print("traj_",k)

        max_delQ = 0
        # setup corresponding realisation of velocity field
        """may have to build the realisation here!!!!"""
        # Vxt = Vx_rzns[k,:,:,:]
        # Vyt = Vy_rzns[k,:,:,:]

        """Jugaad"""
        Vxt = Vx_rzns[k,:,:]
        Vyt = Vy_rzns[k,:,:]


        # for all trajectories in the list of paths
        trajectory = paths[0,k]
        state_traj = []
        coord_traj = []

        test_trajx = []
        test_trajy = []

        #*********ASSUMING THAT 5DT IN TRAJ DATA IS 1 SECOND********
        # s_t = 1
        s_i = None
        s_j = None


        for j in range(0, len(trajectory) - 1, sampling_inerval):  # the len '-1' is to avoid reading NaN at the end of path data
            s_i, s_j = compute_cell(grid, trajectory[j])

            # state_traj.append((s_t, s_i, s_j))
            # coord_traj.append((grid.ts[s_t],trajectory[j][0], trajectory[j][1]))
            state_traj.append((s_i, s_j))
            coord_traj.append((trajectory[j][0], trajectory[j][1]))

            # test_trajx.append(trajectory[j][0])
            # test_trajy.append(trajectory[j][1])
            # s_t+=1

        # if the last sampled point is not endpoint of trajectory, include it in the state/coord_traj
        # s_i_end, s_j_end = compute_cell(grid, trajectory[-2])
        # if (s_i, s_j) != (s_i_end, s_j_end):
        #     state_traj.append((s_t, s_i, s_j))
        #     coord_traj.append((grid.ts[s_t], trajectory[-2][0], trajectory[-2][1]))
        #     test_trajx.append(trajectory[-2][0])
        #     test_trajy.append(trajectory[-2][1])
        #Reverse trajectory orders
        state_traj.reverse()
        coord_traj.reverse()
        test_trajx.reverse()
        test_trajy.reverse()


        # since traj data does not contain start point info, adding it explicitly
        # p, m, n = grid.start_state

        m, n = grid.start_state
        x0 = grid.xs[n]
        y0 = grid.ys[grid.ni - 1 - m]
        state_traj.append(grid.start_state)
        # coord_traj.append((grid.ts[p],x0,y0))
        coord_traj.append((x0,y0))

        # test_trajx.append(x0)
        # # test_trajy.append(y0)
        # if k%500==0:
        #     plt.plot(test_trajx, test_trajy, '-o')
        #Update Q values based on state and possible actions

        for i in range(len(state_traj)-1):
            s1=state_traj[i+1]
            s2=state_traj[i]
            # t ,m,n=s1
            m, n = s1
            p1=coord_traj[i+1]
            p2=coord_traj[i]
            """COMMENTING THIS STATEMENT BELOW"""
            # if (s1[1],s1[2])!=(s2[1],s2[2]):

            #vx=Vxt[t,i,j]
            a1 = Calculate_action(s1,p1,p2, Vxt, Vyt, grid)
            # print("EstQ: YO")
            if (s1[0],s1[1])!= grid.endpos:

                N[s1][a1] += N_inc
                alpha1 = ALPHA / N[s1][a1]

                #update Q considering a1 was performed
                grid.set_state(s1,xcoord=p1[0], ycoord=p1[1])
                r1 = grid.move_exact(a1, Vxt[m,n], Vyt[m,n])
                q_s_a1 = r1
                next_s = grid.current_state()

                if (next_s[0], next_s[1]) != grid.endpos:
                    _, val = max_dict(Q[next_s])
                    q_s_a1 = r1 + val

                old_qsa = Q[s1][a1]
                Q[s1][a1] += alpha1*(q_s_a1 - Q[s1][a1])

                if np.abs(old_qsa - Q[s1][a1]) > max_delQ:
                    max_delQ = np.abs(old_qsa - Q[s1][a1])

        max_delQ_list.append(max_delQ)

    return Q, max_delQ_list


def Learn_policy_from_data(paths, g, Q, N, Vx_rzns, Vy_rzns, num_of_paths=10, num_actions =36, ALPHA=0.5, method = 'reverse_order', num_passes = 1):


    sampling_interval = SAMPLING_Interval
    # Q = EstimateQ_with_parallel_trajs(paths, g, pos_const, sampling_interval, Q, N, Vx, Vy, num_of_paths)
    # Q, max_Qdel_list= EstimateQ_mids_mids2(paths, g, Q, N, Vx_rzns, Vy_rzns, num_of_paths, num_actions, ALPHA, sampling_interval )
    Q, max_Qdel_list= learn_Q_from_trajs(paths, g, Q, N, Vx_rzns, Vy_rzns, num_of_paths, num_actions, ALPHA, sampling_interval, method = method, num_passes= num_passes)
    #Compute policy
    policy=initialise_policy(g)
    for s in Q.keys():
        newa, _ = max_dict(Q[s])
        policy[s] = newa

    return Q, policy, max_Qdel_list




