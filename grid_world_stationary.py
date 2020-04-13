import itertools
from utils.custom_functions import calculate_reward_const_dt
import numpy as np
import math

Pi = math.pi

class Grid:
    def __init__(self, xs, ys, vxs, vys, dt, nt, start, end):
        self.nj = len(xs)
        self.ni = len(ys)
        self.nt = nt
        # print("shapes=", len(xs), len(ys))

        self.dj = np.abs(xs[1] - xs[0])
        self.di = np.abs(ys[1] - ys[0])
        self.dt = dt
        # print("diffs=", self.dj, self.di)

        # state discretizations
        self.xs = xs
        self.ys = ys
        self.vxs = vxs
        self.vys = vys

        # state variables
        self.x = xs[start[1]]
        self.y = ys[self.ni - 1 - start[0]]
        self.vx = None
        self.vy = None

        # i, j , start and end store indices!! dvx, dvy are discrete versions of vx and vy
        # IMPORTANT: while i and j are indices; dvx and dvy are not. they are values from the list vxs and vys
        self.t = 0
        self.i = int(start[0])
        self.j = int(start[1])
        self.dvx = None
        self.dvy = None

        self.endpos = end
        self.startpos = start
        self.start_state = (start[0], start[1])
        # self.edge_states = self.edge_states()

        self.r_outbound = -100
        self.r_terminal = 100
        self.r_otherwise = calculate_reward_const_dt
        self.reward_structure = ['oubound_penalty = '+ str(self.r_outbound), 'Terminal_Reward =' + str(self.r_terminal), 'General_reward: ' + self.r_otherwise.__name__]

    # Rewards and Actions to be dictionaries
    def set_AR(self, Actions):
        self.actions = Actions
        # self.rewards= Rewards

    def discretize(self, v, vs):
        """
        Works only if elements of vs are spaced at regular intervals
        :param v: vx or vy from a cell from a particular realisation of the velocity field
        :param vs: list of vxs or vys- these are discrete possibilites of vxs or vys
        :return: closest element of vx/vy to v
        """
        if v > vs[-1]:
            return vs[-1]
        if v < vs[0]:
            return vs[0]

        del_vs = vs[1] - vs[0]
        # print("in discretize(): v, vs[0], del_vs", v, vs[0], del_vs)
        id = int( (v-vs[0])//del_vs )
        rem = v % del_vs
        if rem > del_vs/2:
            id += 1

        return vs[id]

    # explicitly set state. state is a tuple of indices(m,n,p)
    def set_state(self, state, xcoord=None, ycoord=None):
        # self.t = state[0]
        self.i = state[0]
        self.j = state[1]
        self.dvx = state[2]
        self.dvy = state[3]

        self.x = self.xs[self.j]
        self.y = self.ys[self.ni - 1 - self.i]
        self.vx = self.dvx
        self.vy = self.dvy

        if xcoord != None and ycoord != None:
            self.x = xcoord
            self.y = ycoord


    def current_state(self):
        return (int(self.i), int(self.j), self.dvx, self.dvy)

    def current_pos(self):
        return (int(self.i), int(self.j))


    # MAY NEED TO CHANGE DEFINITION
    def is_terminal(self):
        # return self.actions[state]==None
        return (self.current_pos() == self.endpos)

    def move_exact(self, action):
        """
        Moves agent based on current state and performed action. Collects relevant reward
        :param action:
        :return: immediate reward
        *variant in obsolete.py:  move_exact_noreward()
        """
        r = 0
        if not self.is_terminal() and self.if_within_actionable_time():
            thrust, angle = action
            s0 = self.current_state()
            # print("check: thrust, angle ", thrust, angle)
            # print("self.x, self.y", self.x,self.y)
            vnetx = (thrust * math.cos(angle)) + (self.dvx)
            vnety = (thrust * math.sin(angle)) + (self.dvy)
            xnew = self.x + (vnetx * self.dt)
            ynew = self.y + (vnety * self.dt)
            # print("xnew, ynew",xnew,ynew)

            # if state happens to go out of of grid, bring it back inside
            if xnew > self.xs[-1]:
                xnew = self.xs[-1]
                r += self.r_outbound
            elif xnew < self.xs[0]:
                xnew = self.xs[0]
                r += self.r_outbound
            if ynew > self.ys[-1]:
                ynew = self.ys[-1]
                r += self.r_outbound
            elif ynew < self.ys[0]:
                ynew = self.ys[0]
                r += self.r_outbound
            # print("xnew, ynew after boundingbox", xnew, ynew)
            # rounding to prevent invalid keys

            self.x = xnew
            self.y = ynew

            remx = (xnew - self.xs[0]) % self.dj
            remy = -(ynew - self.ys[-1]) % self.di
            xind = (xnew - self.xs[0]) // self.dj
            yind = -(ynew - self.ys[-1]) // self.di


            # print("rex,remy,xind,yind", remx,remy,xind,yind)
            if remx >= 0.5 * self.dj and remy >= 0.5 * self.di:
                xind += 1
                yind += 1
            elif remx >= 0.5 * self.dj and remy < 0.5 * self.di:
                xind += 1
            elif remx < 0.5 * self.dj and remy >= 0.5 * self.di:
                yind += 1
            #print("rex,remy,xind,yind after upate", remx, remy, xind, yind)
            # print("after update, (i,j)", (yind,xind))
            if not (math.isnan(xind) or math.isnan(yind)):
                self.i = int(yind)
                self.j = int(xind)
                if self.if_edge_state((self.i, self.j)):
                    r += self.r_outbound

            #---- IMPORTANT------
            # We dont know what velocity value it will take on. vx,vy agent will observe in the next position
            # dvx and dvy will be assigned specifically when the agent reaches the next postion
            self.dvx = None
            self.dvy = None

            s_new = self.current_state()
            r += self.r_otherwise(self.dt, self.xs, self.ys, s0, s_new, vnetx, vnety, action)

            if self.is_terminal():
                r += self.r_terminal

        return r


    # !! time to mentioned by index !!
    def ac_state_space(self, only_pos = False, time=None):
        a=set()
        if only_pos == False:
            for i in range(self.ni):
                for j in range(self.nj):
                    if ((i,j)!=self.endpos and not self.if_edge_state((i,j)) ):# does not include states with pos as endpos
                        for dvx in self.vxs:
                            for dvy in self.vys:
                                a.add((i, j, dvx, dvy))
        else:
            for i in range(self.ni):
                for j in range(self.nj):
                    if ((i,j)!=self.endpos and not self.if_edge_state((i,j)) ):# does not include states with pos as endpos
                        a.add((i, j))
        return sorted(a)


    def state_space(self):
        a = set()
        for i in range(self.ni):
            for j in range(self.nj):
                for dvx in self.vxs:
                    for dvy in self.vys:
                        a.add((i,j,dvx,dvy))
        return sorted(a)


    def if_within_time(self):
        return (self.t >= 0 and self.t < self.nt)


    def if_within_actionable_time(self):
        return (self.t >= 0 and self.t < self.nt - self.dt)


    def if_within_TD_actionable_time(self):
        return (self.t >= 0 and self.t < self.nt - 2*self.dt)


    def if_within_grid(self,s):
        i=s[0]
        j=s[1]
        return (j<=(self.nj) -1 and j>=0 and i<=(self.ni)-1 and i>=0)

    def if_edge_state(self, s):
        """
        returns True if state is at the edge
        :param s:
        :return:
        """
        if (s[0] == 0) or (s[0] == self.ni - 1) or (s[1] == 0) or (s[1] == self.nj - 1):
            return True
        else:
            return False

"""Class ends here"""



def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def timeOpt_grid(xs, ys, vxs, vys, dt, nt, F, startpos, endpos, num_actions=36):
    g = Grid(xs, ys, vxs, vys, dt, nt, startpos, endpos)

    # define actions and rewards
    Pi = math.pi
    # speeds in m/s
    speed_list = [F]  # speeds except zero
    angle_list = []

    for i in range(num_actions):
        angle_list.append(round(i * 2 * Pi / num_actions ,14))

    action_list = list(itertools.product(speed_list, angle_list))

    # set actions for grid
    g.set_AR(action_list)

    return g
