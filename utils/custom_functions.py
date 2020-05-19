import numpy as np
import math
from scipy import integrate
import random
import os
from scipy import interpolate
from scipy.optimize import fsolve
import pickle
import csv
from definition import c_r2

Pi =math.pi

def permute_Y(xs, ys):
    P = np.zeros((len(ys), len(xs)))
    j = len(xs)
    for i in range(len(ys)):
        j = j - 1
        P[i, j] = 1
    return P


def my_meshgrid(x, y):
    x = list(x)
    xm = []
    for i in range(len(x)):
        xm.append(x)
    xm = np.asarray(xm)

    y = list(y)
    y.reverse()
    ym = []
    for i in range(len(y)):
        ym.append(y)
    ym = np.asarray(ym)

    return xm, ym.T

def get_angle_in_0_2pi(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

def calculate_reward_const_dt(dt, xs, ys, so, sn, vnet_x, vnet_y, a, degree = False):

    if (so == sn):
        dt_new = dt
    else:
        ni = len(ys)

        io = so[1]
        jo = so[2]
        xo = xs[jo]
        yo = ys[ni - 1 - io]

        inw = sn[1]
        jnw = sn[2]
        xnw = xs[jnw]
        ynw = ys[ni - 1 - inw]
        conv_factor=1
        if degree==True:
            conv_factor=11000
        h = conv_factor*( ((xnw - xo) ** 2 + (ynw - yo) ** 2) ** 0.5 )

        r1 = h/((vnet_x ** 2 + vnet_y ** 2) ** 0.5)

        theta1 = get_angle_in_0_2pi(math.atan2(vnet_y, vnet_x))
        theta2 = get_angle_in_0_2pi(math.atan2(ynw- yo, xnw -xo))
        # print("thetas = ",theta1, theta2)
        theta = np.abs(theta1 - theta2)

        r2 = c_r2*np.abs(math.sin(theta))
        dt_new = r1 + r2
        # print("r1= ", r1, "r2= ", r2)
    return -dt_new


def calculate_reward_var_dt(dt, xs, ys, so, sn, xp, yp, Vx, Vy):
    if (Vx == 0 and Vy == 0) or (so == sn):
        dt_new = dt
    else:
        ni = len(ys)

        io = so[1]
        jo = so[2]
        xo = xs[jo]
        yo = ys[ni - 1 - io]

        inw = sn[1]
        jnw = sn[2]
        xnw = xs[jnw]
        ynw = ys[ni - 1 - inw]

        # print((xo, yo), (xp, yp), (xnw, ynw))
        A = B = C = 0
        # distance of sn (new state) from line
        if xnw - xo == 0:
            d = xp - xnw
        else:
            # line Ax+ By + C =0 line along Vnet
            A = (yp - yo) / (xp - xo)
            B = -1
            C = yo - (xo * (yp - yo) / (xp - xo))

            d = (A * xnw + B * ynw + C) / (A ** 2 + B ** 2) ** 0.5

        # distance from so to sn
        h = ((xnw - xo) ** 2 + (ynw - yo) ** 2) ** 0.5

        # print("d", d, "h", h)

        # length of tangent

        tan_len = (h ** 2 - d ** 2) ** 0.5
        if h ** 2 < d ** 2:
            print("----------------------BUG HAI ------------------------------")
            print("so,sn=", so, sn)
            print("(xo,yo)=", (xo, yo))
            print("(xnw,ynw)=", (xnw, ynw))
            print("(xp,yp=", (xp, yp))
            print("A, B, C =", A, B, C)
            print("h=", h)
            print("d=", d)
            print("tan_len=", tan_len)
        # actual distance travelled in dT
        tot_dis = ((xp - xo) ** 2 + (yp - yo) ** 2) ** 0.5
        # print("tan_len=", tan_len, "tot_dis", tot_dis)

        err = tot_dis - tan_len
        # print(err)

        dt_new = dt - (err / ((Vx ** 2 + Vy ** 2) ** 0.5))

    return -dt_new


"""

The functions below are used in the stochatic version of the DP 
Problem

"""


# state transition probability derived from prob dist of velocities
def p_sas(xi, xf, yi, yf, sigx, sigy):
    def pdf(x, y):
        ux = 0
        uy = 0
        return np.exp(- ((((x - ux) ** 2) / (2 * (sigx ** 2))) + (((y - uy) ** 2) / (2 * (sigy ** 2))))) / (
                2 * math.pi * sigx * sigy)

    return integrate.nquad(pdf, [[xi, xf], [yi, yf]])[0]


# generates matrix with samples from normal curve. Used to simulate uncertainty in velocity field
def gaussian_matrix(t, r, c, u, sigma):
    a = np.zeros((t, r, c))
    for k in range(t):
        for i in range(r):
            for j in range(c):
                a[k, i, j] = np.random.normal(u, sigma)
    return a


# finds number of states s' on either sides where p_sas is >thresh , i.e. significant
def find_mn(dx, dy, sigx, sigy, thresh):
    i = 0
    j = 0
    val = 1
    while val > thresh:
        val = p_sas((j - 0.5) * dx, (j + 0.5) * dx, (i - 0.5) * dy, (i + 0.5) * dy, sigx, sigy)
        j = j + 1
    m = j - 1

    val = 1
    i = 0
    j = 0
    while val > thresh:
        val = p_sas((j - 0.5) * dx, (j + 0.5) * dx, (i - 0.5) * dy, (i + 0.5) * dy, sigx, sigy)
        i = i + 1
    n = i - 1
    return m, n


def transition_probs(m, n, dx, dy, sigx, sigy, thresh):
    psas_dict = {}
    for i in range(-n, n + 1, 1):
        for j in range(-m, m + 1, 1):
            prob = p_sas((j - 0.5) * dx, (j + 0.5) * dx, (i - 0.5) * dy, (i + 0.5) * dy, sigx, sigy)
            if prob > thresh:
                psas_dict[(i, j)] = prob
    return psas_dict, psas_dict.keys()


def stoch_vel_field(d_vStream, xm, ym, cov_sigma, coef):
    t, r, c = d_vStream.shape
    st_vStream=np.zeros((t,r,c))
    fxgrd=np.ndarray.flatten(xm)
    fygrd=np.ndarray.flatten((ym))
    for k in range(t):
        fl_vel=np.ndarray.flatten(d_vStream[k,:,:])
        l=len(fl_vel)
        cov=np.zeros((l,l))
        for i in range(l):
            for j in range(l):
                rsqm=(fxgrd[i]-fxgrd[j])**2 + (fygrd[i]-fygrd[j])**2
                cov[i,j]=coef*np.exp(-rsqm/(2*cov_sigma**2))
        fl_st_vel=np.random.multivariate_normal(fl_vel,cov)
        st_vStream[k,:,:]=fl_st_vel.reshape((r,c))
    return st_vStream


def boltzman_stoch_action(Q,s,g,T):
    prob={}
    sum_of_probs=0
    for a in g.actions:
        sum_of_probs += math.exp(Q[s][a]/T)
    for a in g.actions:
        prob[a]=(math.exp(Q[s][a]/T))/sum_of_probs

    #generate intervals over (0,1) for actions
    intervals={}
    i=0
    first=True
    a_old=None
    for a in g.actions:
        if first:
            first=False
            intervals[a]=(0,prob[a])
        else:
            intervals[a] = (prob[a_old], prob[a] + prob[a_old])
        a_old = a

    #checks if a given value is in the range (a,b)
    def check_if_in_range(value, Range):
        if value>=Range[0] and value<=Range[1]:
            return True

    #sample action from prob distribution
    rand_num=np.random.random()
    for action, interval_range in intervals:
        if check_if_in_range(rand_num, interval_range):
            return action



def stochastic_action_eps_greedy(policy, s, g, eps, Q=None):
    p = np.random.random()
    n = len(g.actions)
    if p < (1 - eps):
        if Q==None:
            newa = policy[s]
        else:
            newa, _ = max_dict(Q[s])
    else:
        #returns a random integer between [0,n-1]
        i = random.randint(0, n - 1)
        newa = g.actions[i]
    return newa



def random_action(a, s, g, eps):
    p = np.random.random()
    n = len(g.actions)
    if p < (1 - eps + (eps / n)):
        newa = a
    else:
        i = random.randint(0, n - 1)
        newa = g.actions[s][i]

    return newa

def initialise_directed_policy(g, action_states):
    policy = {}

    Pi = math.pi
    for s in action_states:
        i, j = s
        # print("start", s)
        i2, j2 = g.endpos
        # print("i2,j2", i2, j2)
        if j2 == j:
            if i2 > i:
                policy[s] = (1, 1.5 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.5 * Pi)
        elif j2 > j:
            if i2 > i:
                policy[s] = (1, 1.75 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.25 * Pi)
            elif i2 == i:
                policy[s] = (1, 0)
        elif j2 < j:
            if i2 > i:
                policy[s] = (1, 1.25 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.75 * Pi)
            elif i2 == i:
                policy[s] = (1, Pi)
        # if i == 9 or j == 9:
        #     print("state, action ", s, policy[s])

    return policy

def initialise_policy(g):
    policy = {}
    n=len(g.actions)
    for s in g.state_space():
        policy[s]=None

    for s in g.ac_state_space():
        i = random.randint(0, n - 1)
        policy[s] = g.actions[i]

    return policy


def initialise_policy_from_initQ(Q):
    policy={}
    for s in Q.keys():
        for a in Q[s].keys():
            besta,_=max_dict(Q[s])
            policy[s]=besta
    return policy



def initialise_Q_N(g, init_Qval, init_Nval):
    Q = {}
    N = {}
    for s in g.ac_state_space():
        Q[s] = {}
        N[s] = {}
        for a in g.actions:
            Q[s][a] = init_Qval
            N[s][a] = init_Nval
    return Q, N


def calc_net_velocity_angle(vx,vy,a):
    F, theta = a
    return get_angle_in_0_2pi( math.atan2(F*math.sin(theta) + vy, F*math.cos(theta) + vx) )

def calc_net_velocity(vx, vy, a):
    F, theta = a
    vy_net = F*math.sin(theta) + vy
    vx_net = F*math.cos(theta) + vx
    mag_v = (vx_net**2 + vy_net**2)**.5
    ang_v = get_angle_in_0_2pi(math.atan2(vy_net, vx_net))
    return (mag_v, ang_v)


def action_angle(traj_theta, actions, vx, vy, F):
    print()
    diff_list = []
    blocked_actions = []
    angle_list = []
    for idx in range(len(actions)):
        angle_list.append(actions[idx][1])
        (net_v_mag, net_v_angle) = calc_net_velocity(vx, vy, actions[idx])
        if net_v_mag < 0.001:
            blocked_actions.append(idx)
        diff_list.append(np.abs(traj_theta - net_v_angle))
    sorted_ids = np.argsort(diff_list)

    min_id = None
    for idx in sorted_ids:
        if idx in blocked_actions:
            pass
        else:
            min_id = idx
            # print("min_id =", min_id)
            break
    action = actions[min_id]
    return action

def Calculate_action(state_1,p1,p2, vx, vy, g):
    traj_theta = get_angle_in_0_2pi(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    F = g.actions[0][0]
    action = action_angle(traj_theta, g.actions, vx, vy, F)
    return action


def initialise_guided_Q_N(g, init_Qval, guiding_Qval,  init_Nval):
    Q = {}
    N = {}
    end_i, end_j = g.endpos
    # p2 = (None, g.xs[end_j], g.ys[g.ni - 1 - end_i])
    p2 = (g.xs[end_j], g.ys[g.ni - 1 - end_i])
    Vx = Vy = np.zeros((len(g.xs), len(g.ys)))


    for s in g.ac_state_space():
        Q[s] = {}
        N[s] = {}
        g.set_state(s)
        # p1 = (None, g.x, g.y)
        p1 = (g.x, g.y)
        m, n= s
        targeting_action = Calculate_action(s, p1, p2, Vx[m,n], Vy[m,n],  g)

        for a in g.actions:

            if a==targeting_action:
                Q[s][a] = guiding_Qval

            else:
                Q[s][a] = init_Qval

            N[s][a] = init_Nval

    return Q, N


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        # print("dict test:", k , v)
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

def writePolicytoFile(policy, filename):
    outputfile = open(filename+'.txt', 'w')
    print(policy, file=outputfile)
    outputfile.close()
    return

def picklePolicy(policy, filename):
    with open(filename +'.p', 'wb') as fp:
        pickle.dump(policy, fp, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickled_File(filename):
    with open(filename +'.p', 'rb') as fp:
        File = pickle.load(fp)
    print("READING " + filename + '.p')
    return File


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def Map_vector_to_grid(u, node_u):
    m, n = node_u.shape
    U = np.zeros((m - 2, n - 2))
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            matlab_ind = node_u[i, j]
            python_ind = matlab_ind - 1
            U[i - 1, j - 1] = u[python_ind, 0]

    return U


# Interpolation of velocites from their grids to XP,YP grid
def interpolate_to_XP_grid(XU, YU, U, XP, YP):
    flat_xu = np.ndarray.flatten(XU)
    flat_yu = np.ndarray.flatten(YU)
    flat_u = np.ndarray.flatten(U)
    U_interpolated = interpolate.griddata((flat_xu, flat_yu), flat_u, (XP, YP))
    return U_interpolated


def append_summary_to_summaryFile(summary_file, summary_data):
    myFile = open('Experiments/Exp_summary.csv', 'a')
    with myFile:
       writer = csv.writer(myFile)
       writer.writerows([summary_data])


def read_n_rearrange_trajSet(filename):
    """
    Rearranges list to be usable in frechet_dist()
    :param filename: filename of pickled traj_set coordinates obtained from plot_all_traj_set()
    :return: rearranged traj_set coords
    """
    old_trajSet =read_pickled_File(filename)
    new_traj_list = []
    print("len(old_trajSet) =", len(old_trajSet))

    for i in range(len(old_trajSet)):
        print("i= ", i)
        P = old_trajSet[i]
        Pnew = []
        print("len(P[0])= ", len(P[0]))
        for j in range(len(P[0])):
            Pnew.append([P[0][j],P[1][j]])
        new_traj_list.append(Pnew)
    return new_traj_list



def calc_mean_and_std(scalar_list):
    """
    calulates mean, variance and None count of entries in given list.
    :param scalar_list: list of scalar and None elements
    :return: mean, std dev, good_cnt, Nonecount
    """

    cnt = 0
    sum_x = 0
    sum_x_sq = 0
    num_rzns = len(scalar_list)

    for i in range(num_rzns):
        if scalar_list[i] != None:
            cnt += 1
            sum_x += scalar_list[i]
            sum_x_sq += scalar_list[i]**2

    if cnt == 0:
        print("\n $$$$------ ERROR: List contains only Nones --------$$$$ \n")

    mean = sum_x/cnt
    var = (sum_x_sq/cnt) - (sum_x/cnt)**2
    std = var**0.5
    none_cnt = num_rzns - cnt

    return mean, std, cnt, none_cnt