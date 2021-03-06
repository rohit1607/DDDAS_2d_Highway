import numpy as np
import matplotlib.pyplot as plt
import math
from definition import c_ni, ROOT_DIR
from utils.custom_functions import createFolder, picklePolicy, read_pickled_File
from os.path import join
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx


def action_to_quiver(a):
    vt = a[0]
    theta = a[1]
    vtx = vt * math.cos(theta)
    vty = vt * math.sin(theta)
    return vtx, vty

"""
def plot_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y, fname=None, lastfig=None):
    # time calculation and state trajectory
    trajectory = []
    xtr = []
    ytr = []
    vtx_list = []
    vty_list = []

    i, j = g.start_state
    # print(t,x,y,vStream_x[t,x,y])

    g.set_state((i, j))
    # print(g.current_state())
    trajectory.append((i, j))
    a = policy[g.current_state()]
    vtx, vty = action_to_quiver(a)
    vtx_list.append(vtx)
    vty_list.append(vty)

    xtr.append(xs[j])
    ytr.append(ys[g.ni - 1 - i])

    # set grid
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    minor_xticks = np.arange(xs[0] - 0.5 * g.dj, xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(ys[0] - 0.5 * g.di, ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(xs[0], xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(ys[0], ys[-1] + 2 * g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')

    plt.quiver(xtr, ytr, vtx_list, vty_list)

    plt.plot(xtr, ytr)
    plt.scatter(xtr, ytr)

    # plots start point
    st_point = trajectory[0]

    plt.scatter(xs[st_point[1]], ys[g.ni - 1 - st_point[0]], c='g')
    # plots current point
    # plt.scatter(xs[trajectory[-1][0]], ys[g.ni-1-trajectory[-1][1]], c='k')
    # plots end point
    plt.scatter(xs[g.endpos[1]], ys[g.ni - 1 - g.endpos[0]], c='r')
    # plots current point
    plt.scatter(xs[j], ys[g.ni - 1 - i])
    plt.quiver(X, Y, vStream_x[:, :], vStream_y[:, :])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    if fname != None and lastfig==None:
        filename = fname + str(t) + ".png"
        plt.savefig(filename)
        plt.close()

    # print("in loop---")
    G = 0
    flag=False

    while not g.is_terminal() and g.if_within_actionable_time() :

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        minor_xticks = np.arange(xs[0] - 0.5 * g.dj, xs[-1] +  2*2*g.dj, g.dj)
        minor_yticks = np.arange(ys[0] - 0.5 * g.di, ys[-1] +  2*g.di, g.di)

        major_xticks = np.arange(xs[0], xs[-1] + 2*g.dj, 5 * g.dj)
        major_yticks = np.arange(ys[0], ys[-1] + 2*g.di, 5 * g.di)

        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(major_xticks)
        ax.set_yticks(major_yticks)

        ax.grid(which='major', color='#CCCCCC', linestyle='')
        ax.grid(which='minor', color='#CCCCCC', linestyle='--')

        # print("state", (t, i, j))
        # print("action", a)
        # print("vfx", vStream_x[t, i, j])
        # print("vfy", vStream_y[t, i, j])

        r = g.move( a, vStream_x[t, i, j], vStream_y[t, i, j] )
        G = G + r
        (t, i, j) = g.current_state()

        # print(g.current_state(), (t, x, y), a, vStream_x[t, x, y], vStream_y[t, x, y])
        trajectory.append((i, j))

        xtr.append(xs[j])
        ytr.append(ys[g.ni - 1 - i])


        if not g.is_terminal() and g.if_within_actionable_time():
            a = policy[g.current_state()]
        else:
            a=None
            flag=True

        if a != None:
            vtx, vty = action_to_quiver(a)
            vtx_list.append(vtx)
            vty_list.append(vty)
            plt.quiver(xtr, ytr, vtx_list, vty_list)
        else:
            plt.quiver(xtr[0:len(xtr) - 1], ytr[0:len(ytr) - 1], vtx_list, vty_list)
        plt.plot(xtr, ytr)
        plt.scatter(xtr, ytr)

        plt.scatter(xs[st_point[1]], ys[g.ni - 1 - st_point[0]], c='g')
        # plt.scatter(xs[trajectory[-1][0]], ys[g.ni - 1 - trajectory[-1][1]], c='k')
        plt.scatter(xs[g.endpos[1]], ys[g.ni - 1 - g.endpos[0]], c='r')
        plt.scatter(xs[j], ys[g.ni - 1 - i])
        plt.quiver(X, Y, vStream_x[t, :, :], vStream_y[t, :, :])
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

        if fname!=None:
            if lastfig==True and flag==True:
                filename = fname + str(t) + ".png"
                plt.savefig(filename)
                plt.close()
            elif lastfig==None:
                filename = fname + str(t) + ".png"
                plt.savefig(filename)
                plt.close()


    return trajectory, (t, i, j), G
"""

def plot_exact_trajectory(g, policy, X, Y, vStream_x, vStream_y, fpath, fname=None, lastfig=None):
    # time calculation and state trajectory
    trajectory = []
    xtr = []
    ytr = []
    vtx_list = []
    vty_list = []

    i, j = g.start_state
    # print(i,j)

    g.set_state((i, j))
    # print(g.current_state())
    trajectory.append((i, j))
    a = policy[g.current_state()]
    vtx, vty = action_to_quiver(a)
    vtx_list.append(vtx)
    vty_list.append(vty)

    xtr.append(g.x)
    ytr.append(g.y)

    #set grid
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2*g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2*g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2*g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2*g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')

    plt.quiver(xtr, ytr, vtx_list, vty_list)

    plt.plot(xtr, ytr)
    plt.scatter(xtr, ytr)

    # plots start point
    st_point = trajectory[0]
    plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], c='g')
    # plots current point
    # plt.scatter(xs[trajectory[-1][0]], ys[g.ni-1-trajectory[-1][1]], c='k')
    # plots end point
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
    # plots current point
    plt.scatter(g.x, g.y)
    plt.quiver(X, Y, vStream_x[:, :], vStream_y[:, :])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    t=0
    if fname != None and lastfig == None:
        filename = fname + str(t) + ".png"
        plt.savefig(join(fpath + filename))
        plt.close()
    # print("in loop---")
    G = 0
    flag=False
    while not g.is_terminal():
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(1, 1, 1)
        minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] +  2*g.dj, g.dj)
        minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] +  2*g.di, g.di)

        major_xticks = np.arange(g.xs[0], g.xs[-1] + 2*g.dj , 5 * g.dj)
        major_yticks = np.arange(g.ys[0], g.ys[-1] + 2*g.di , 5 * g.di)

        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(major_xticks)
        ax.set_yticks(major_yticks)

        ax.grid(which='major', color='#CCCCCC', linestyle='')
        ax.grid(which='minor', color='#CCCCCC', linestyle='--')
        # print("***in plot functions***")
        # print("state", (i, j))
        # print("action", a)
        # print("vfx", vStream_x[i, j])
        # print("vfy", vStream_y[i, j])
        # print("---------------")
        # print('before')
        # print("in plot: check coords:", g.x, g.y)
        # print("in plot: check states:", g.i, g.j)
        # print("policty", a)
        # print("current: ",vStream_x[i, j], vStream_y[i, j] )
        r = g.move_exact(a, vStream_x[i, j], vStream_y[i, j])
        G = G + r
        (i, j) = g.current_state()

        # print(g.current_state(), (t, x, y), a, vStream_x[t, x, y], vStream_y[t, x, y])
        trajectory.append((i, j))

        xtr.append(g.x)
        ytr.append(g.y)
        # print("in plot: check coords:", g.x, g.y)
        # print("in plot: check states:", g.i, g.j)

        if not g.is_terminal():
            a = policy[g.current_state()]
        else:
            a = None
            flag = True

        # print(t)
        if t>g.ni*c_ni:
            flag=True

        if a != None:
            vtx, vty = action_to_quiver(a)
            vtx_list.append(vtx)
            vty_list.append(vty)
            plt.quiver(xtr, ytr, vtx_list, vty_list)
        else:
            plt.quiver(xtr[0:len(xtr) - 1], ytr[0:len(ytr) - 1], vtx_list, vty_list)

        # plt.plot(xtr, ytr)
        # plt.scatter(xtr, ytr)
        #
        # plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], c='g')
        # # plt.scatter(xs[trajectory[-1][0]], ys[g.ni - 1 - trajectory[-1][1]], c='k')
        # plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
        # plt.scatter(g.x, g.y)
        # plt.quiver(X, Y, vStream_x[:, :], vStream_y[:, :])
        # plt.grid()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        if fname != None:
            if lastfig == True and flag == True:
                plt.plot(xtr, ytr)
                plt.scatter(xtr, ytr)

                plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], c='g')
                # plt.scatter(xs[trajectory[-1][0]], ys[g.ni - 1 - trajectory[-1][1]], c='k')
                plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
                plt.scatter(g.x, g.y)
                plt.quiver(X, Y, vStream_x[:, :], vStream_y[:, :])
                plt.grid()
                plt.gca().set_aspect('equal', adjustable='box')

                filename = fname + str(t) + ".png"
                plt.savefig(join(fpath,filename), dpi=300)
                # plt.show()
                plt.close()
                break
            elif lastfig == None:
                filename = fname + str(t) + ".png"
                plt.savefig(join(fpath, filename))
                plt.close()

        t+=1
    return trajectory, G


def plot_learned_policy(g, DP_params = None, QL_params = None,  showfig = False):
    """
    Plots learned policy
    :param g: grid object
    :param DP_params: [policy, filepath]
    :param QL_params: [policy, Q, init_Q, label_data, filepath]
    :param showfig: whether you want to see fig during execution
    :return:
    """
    """
    QL_params:
    :param Q: Leared Q against which policy is plotted. This is needed just for a check in the QL case. TO plot policy only at those states which have been updated
    :param policy: Learned policy.
    :param init_Q: initial value for Q. Just like Q, required only for the QL policy plot
    :param label_data: Labels to put on fig. Currently requiered only for QL
    """
    full_file_path = ROOT_DIR
    if DP_params == None and QL_params == None:
        print("Nothing to plot! Enter either DP or QL params !")
        return

    # set grid
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(1, 1, 1)

    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax1.set_xticks(minor_xticks, minor=True)
    ax1.set_yticks(minor_yticks, minor=True)
    ax1.set_xticks(major_xticks)
    ax1.set_yticks(major_yticks)

    ax1.grid(which='major', color='#CCCCCC', linestyle='')
    ax1.grid(which='minor', color='#CCCCCC', linestyle='--')
    xtr=[]
    ytr=[]
    ax_list=[]
    ay_list=[]


    if QL_params != None:
        policy, Q, init_Q, label_data, full_file_path = QL_params
        F, ALPHA, initq, QIters = label_data
        ax1.text(0.1, 9, 'F=(%s)'%F, fontsize=12)
        ax1.text(0.1, 8, 'ALPHA=(%s)'%ALPHA, fontsize=12)
        ax1.text(0.1, 7, 'initq=(%s)'%initq, fontsize=12)
        ax1.text(0.1, 6, 'QIters=(%s)'%QIters, fontsize=12)
        for s in Q.keys():
            # for a in Q[s].keys():
                # if s[0]==0 and a == policy[s]: # to print policy at time t = 0
            a = policy[s]
            if Q[s][a] != init_Q: # to plot policy of only updated states
                # t,i,j = s
                i, j = s
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                # print("test", s, a_policy)
                ax, ay = action_to_quiver(a)
                ax_list.append(ax)
                ay_list.append(ay)
                # print(i,j,g.xs[j], g.ys[g.ni - 1 - i], ax, ay)

    if DP_params != None:
        policy, full_file_path = DP_params
        for s in g.ac_state_space():
            a = policy[s]
            i, j = s
            xtr.append(g.xs[j])
            ytr.append(g.ys[g.ni - 1 - i])
            # print("test", s, a_policy)
            ax, ay = action_to_quiver(a)
            ax_list.append(ax)
            ay_list.append(ay)

    plt.quiver(xtr, ytr, ax_list, ay_list)

    ax1.scatter(g.xs[g.start_state[1]], g.ys[g.ni - 1 - g.start_state[0] ], c='g')
    ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
    if showfig ==True:
        plt.show()

    fig1.savefig(full_file_path+'_policy_plot.png', dpi=300)

    return


def plot_all_policies(g, Q, policy, init_Q , label_data, showfig = False, Iters_after_update=None, full_file_path= None):

    createFolder(full_file_path)

    # set grid
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(1, 1, 1)

    F, stream_speed, ALPHA, initq, QIters = label_data

    ax1.text(0.1, 9, 'F=(%s)' % F, fontsize=12)
    ax1.text(0.1, 8, 'ALPHA=(%s)' % ALPHA, fontsize=12)
    ax1.text(0.1, 7, 'initq=(%s)' % initq, fontsize=12)
    ax1.text(0.1, 6, 'QIters=(%s)' % QIters, fontsize=12)

    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax1.set_xticks(minor_xticks, minor=True)
    ax1.set_yticks(minor_yticks, minor=True)
    ax1.set_xticks(major_xticks)
    ax1.set_yticks(major_yticks)

    ax1.grid(which='major', color='#CCCCCC', linestyle='')
    ax1.grid(which='minor', color='#CCCCCC', linestyle='--')
    ax1.scatter(g.xs[g.start_state[2]], g.ys[g.ni - 1 - g.start_state[1]], c='g')
    ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')

    xtr = []
    ytr = []
    ax_list = []
    ay_list = []

    for s in Q.keys():
        t, i, j = s
        for a in Q[s].keys():
            if Q[s][a] != init_Q and a==policy[s]:
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                # print("test", s, a_policy)
                ax, ay = action_to_quiver(a)
                ax_list.append(ax)
                ay_list.append(ay)
                # print(i,j,g.xs[j], g.ys[g.ni - 1 - i], ax, ay)


    plt.quiver(xtr, ytr, ax_list, ay_list)
    filename= full_file_path +'/policy@t'
    fig1.savefig(filename, dpi=300)
    if showfig == True:
        plt.show()
    return


def plot_max_Qvalues(Q,policy, XP,YP):
    m,n = XP.shape
    Z=np.zeros((m,n))
    for s in Q.keys():
        a = policy[s]
        # t,i,j = s
        i,j =s
        Z[i,j]= Q[s][a]

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = plt.axes(projection="3d")
    mycmap = plt.get_cmap('coolwarm')

    ax.plot_surface(XP, YP, Z, cmap=mycmap, linewidth=0, antialiased=False)
    plt.show()



def plot_exact_trajectory_set(g, policy, X, Y, vStream_x, vStream_y, fpath, fname='Trajectories'):
    # time calculation and state trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # set grid
    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')
    st_point= g.start_state
    plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], c='g')
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.quiver(X, Y, vStream_x[0, :, :], vStream_y[0, :, :])

    n_rzn,m,n = vStream_x.shape
    bad_count =0
    t_list=[]
    G_list=[]
    traj_list = []
    for rzn in range(n_rzn):
        g.set_state(g.start_state)
        dont_plot =False
        t = 0
        G = 0

        xtr = []
        ytr = []

        i, j = g.start_state

        a = policy[g.current_state()]
        xtr.append(g.x)
        ytr.append(g.y)

        # while not g.is_terminal() and g.if_within_actionable_time() and g.current_state:
        while True:
            r = g.move_exact(a, vStream_x[rzn, i, j], vStream_y[rzn, i, j])
            G = G + r
            t += 1
            (i, j) = g.current_state()

            xtr.append(g.x)
            ytr.append(g.y)

            if g.if_edge_state((i,j)):
                bad_count += 1
                dont_plot=True
                break

            if (not g.is_terminal()):
                a = policy[g.current_state()]
            else:
                break

            if t > g.ni * c_ni: #if trajectory goes haywire, dont plot it.
                bad_count+=1
                dont_plot=True
                break

        if dont_plot==False:
            plt.plot(xtr, ytr)
            traj_list.append((xtr,ytr))
            t_list.append(t)
            G_list.append(G)

        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list.append(None)
            G_list.append(None)


    if fname != None:
        plt.savefig(join(fpath,fname), dpi=300)
        print("*** pickling traj_list ***")
        picklePolicy(traj_list, join(fpath,fname))
        print("*** pickled ***")

    return t_list, G_list, bad_count



def plot_max_delQs(max_delQ_list_1, filename= None ):
     fig = plt.figure(figsize=(10, 10))
     ax = fig.add_subplot(1, 1, 1)
     plt.plot(max_delQ_list_1)

     if filename!= None:
         plt.savefig(filename, dpi=300)

     return


def plot_input_trajectory(paths, idx, g, X, Y, vStream_x, vStream_y,
                                                fname='GivenTrajectories'):
    # time calculation and state trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # set grid
    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.set_xlim([15,85])
    ax.set_ylim([15,85])
    ax.grid(which='major', color='#CCCCCC', linestyle='', zorder = -1)
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', zorder = -1)
    st_point= g.start_state
    plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], c='g', s =300, zorder = 100)
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r', s = 300, zorder =100)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.quiver(X, Y, vStream_x[idx, :, :], vStream_y[idx, :, :])

    trajectory = paths[0, idx]
    plt.plot(trajectory[0:-5,0], trajectory[0:-5,1], linewidth = 8, zorder= 99)
    # plt.show()
    if fname != None:
         plt.savefig(fname+str(idx), dpi=300)

    return


def plot_paths_colored_by_EAT(plotFile=None, baseFile=None, savePath_fname=None):
    msize = 15
    fsize = 3

    #---------------------------- beautify plot ---------------------------
    # time calculation and state trajectory
    fig = plt.figure(figsize=(fsize, fsize))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # set grid

    minor_ticks = [i for i in range(101) if i % 20 != 0]
    major_ticks = [i for i in range(0, 120, 20)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(b=True, which='both', color='#CCCCCC', axis='both', linestyle='-', alpha=0.5)
    ax.tick_params(axis='both', which='both', labelsize=6)

    ax.set_xlabel('X (Non-Dim)')
    ax.set_ylabel('Y (Non-Dim)')

    #     st_point= g.start_state
    #     plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    #     plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
    plt.gca().set_aspect('equal', adjustable='box')




    #---------------------------- main plot ---------------------------
    # read file
    plot_set = read_pickled_File(plotFile)

    # calculate time
    time_list = []
    l = len(plot_set)

    # if baseFile is provided, comparison plot will be made. colorbar will show EAT time differnces.
    if baseFile != None:
        base_traj_set = read_pickled_File(baseFile)
        l_base = len(base_traj_set)
        # sanity check
        if l != l_base:
            print("ERROR: Unfair Comparison. Two lists should have data across same number of realisations")
            return

        for i in range(l):
            if plot_set[i] != None and base_traj_set[i] != None:
                t_plot_set_i = len(plot_set[i][0])
                t_base_set_i = len(base_traj_set[i][0])
                time_list.append(t_plot_set_i - t_base_set_i)

    # if baseFile is NOT provided, then the basePlot data will be plotted.
    else:
        for i in range(l):
            if plot_set[i] != None:
                time_list.append(len(plot_set[i][0]))

    # set colormap
    jet = cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=np.min(time_list), vmax=np.max(time_list))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    scalarMap._A = []

    # plot plot_set
    for i in range(int(l)):
        if plot_set[i] != None:
            colorval = scalarMap.to_rgba(time_list[i])
            plt.plot(plot_set[i][0], plot_set[i][1], color=colorval, alpha=0.6)
    plt.colorbar(scalarMap)

    if savePath_fname != None:
        plt.savefig(savePath_fname, bbox_inches="tight", dpi=300)

    plt.show()

    return time_list