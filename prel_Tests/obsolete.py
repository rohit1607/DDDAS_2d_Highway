"""from grid_world_stationary.py
    move function without assigning rewards
"""
# def move_exact_noreward(self, action, Vx, Vy):
#     # if math.isnan(Vx):
#     #     Vx = 0
#     # if math.isnan(Vy):
#     #     Vy = 0
#
#     if self.is_terminal() == False and self.if_within_actionable_time():
#         thrust, angle = action
#         # print("check: thrust, angle ", thrust, angle)
#         # print("self.x, self.y", self.x,self.y)
#         vnetx = (thrust * math.cos(angle)) + (Vx)
#         vnety = (thrust * math.sin(angle)) + (Vy)
#         xnew = self.x + (vnetx * self.dt)
#         ynew = self.y + (vnety * self.dt)
#         # print("xnew, ynew",xnew,ynew)
#
#         if xnew > self.xs[-1]:
#             xnew = self.xs[-1]
#         elif xnew < self.xs[0]:
#             xnew = self.xs[0]
#         if ynew > self.ys[-1]:
#             ynew = self.ys[-1]
#         elif ynew < self.ys[0]:
#             ynew = self.ys[0]
#         # print("xnew, ynew after boundingbox", xnew, ynew)
#         # rounding to prevent invalid keys
#
#         self.x = xnew
#         self.y = ynew
#
#         remx = (xnew - self.xs[0]) % self.dj
#         remy = -(ynew - self.ys[-1]) % self.di
#         xind = (xnew - self.xs[0]) // self.dj
#         yind = -(ynew - self.ys[-1]) // self.di
#
#         # print("rex,remy,xind,yind", remx,remy,xind,yind)
#
#         if remx >= 0.5 * self.dj and remy >= 0.5 * self.di:
#             xind += 1
#             yind += 1
#         elif remx >= 0.5 * self.dj and remy < 0.5 * self.di:
#             xind += 1
#         elif remx < 0.5 * self.dj and remy >= 0.5 * self.di:
#             yind += 1
#         # print("rex,remy,xind,yind after upate", remx, remy, xind, yind)
#         # print("after update, (i,j)", (yind,xind))
#         if not (math.isnan(xind) or math.isnan(yind)):
#             self.i = int(yind)
#             self.j = int(xind)
#
#     return
#
#
# def R(self, s, snew):
#
#     if snew == self.endpos:
#         return +10
#
#     elif (snew in self.edge_states):
#         return -100
#
#     else:
#         return -1






"""
from build_Q_from_Trajs.py
variant of build_experience_buffer()

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


"""
from build_model.py
2 state variant of compute_transition_probability_and_rewards()
"""

#def compute_transition_probability_and_rewards(transition_dict, g, num_rzns, Vx_rzns, Vy_rzns):
    s_count = 0
    for s in state_list:
        s_count += 1
        i0, j0 = s
        if s_count%100 == 0:
            print("s_count: ", s_count)
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
