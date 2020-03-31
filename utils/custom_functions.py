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

        io = so[0]
        jo = so[1]
        xo = xs[jo]
        yo = ys[ni - 1 - io]

        inw = sn[0]
        jnw = sn[1]
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


def calc_action_angle(traj_theta, vx, vy, F,g):
    def solve_eqn_1(z):
        b = math.tan(traj_theta)
        func_val = (b * F * math.cos(z)) - (F * math.sin(z)) - vy + (b * vx)
        return func_val

    def solve_eqn_2(z):
        b = math.tan(traj_theta)
        func_val = (b * F * math.cos(z)) - (F * math.sin(z)) - vy + (b * vx)
        return func_val

    zGuess = traj_theta
    z = fsolve(solve_eqn_1, zGuess)
    angle = z[0]
    #if angle is negative, make it positive
    while angle<0:
        angle += 2*Pi
    #if angle is grater than 2Pi, bring it beween [0, 2Pi]
    angle = z[0] % (2 * Pi)

    # print("in calc_action_angle : ",angle)
    if np.abs(F*math.cos(angle) + vx) < 0.001:
        print("denominator: ", np.abs(F*math.cos(angle) + vx))
        diff_list= []
        for a in g.actions:
            net_angle = calc_net_velocity_angle(vx,vy,a)
            diff_list.append(np.abs(traj_theta -net_angle))
        idx = np.argmin(np.array(diff_list))
        angle = g.actions[idx][1]

    return angle


def Calculate_action(state_1,p1,p2, vx, vy, g):
    """Handles only single speed"""
    # traj_theta =  math.atan2(p2[2] - p1[2], p2[1] - p1[1])
    # t, m, n = state_1
    traj_theta = get_angle_in_0_2pi(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

    """Will have to be changed (include time) for the general problem!!!"""
    F = g.actions[0][0]
    ac_theta = calc_action_angle(traj_theta, vx, vy, F, g)

    #angles discretised at
    num_actions = len(g.actions)
    d_theta = 2*Pi/num_actions
    l = int(ac_theta//d_theta)
    a1 =g.actions[l]
    a2 = g.actions[(l+1)%num_actions]
    a1_angle = a1[1]
    a2_angle = a2[1]

    # if a2_angle == 2*Pi:
    #     a2_angle=0
    # if flag==True:
    #     print("angles", a1_angle,a2_angle)

    del_a1_angle = np.abs(ac_theta - a1_angle)
    del_a2_angle = np.abs(ac_theta - a2_angle)
    if del_a1_angle > Pi:
        del_a1_angle -= Pi
    if del_a2_angle > Pi:
        del_a2_angle -= Pi

    if del_a1_angle<=del_a2_angle:
        best_action = a1
    else:
        best_action = a2

    if state_1[0] == 40 and state_1[1] >= 58 and state_1[1] <= 70:
        if ac_theta > Pi:
            print("Check: ",state_1, p1, p2, traj_theta, ac_theta, a1, a2, del_a1_angle, del_a2_angle)

    return best_action


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






