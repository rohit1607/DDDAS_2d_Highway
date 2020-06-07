import os
# Project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# increment N[s][a] by N_inc each time (s,a) visited
N_inc = 0.005

#Take every nth point from a trajectory waypoint list
#the value is 4 for 3d case because dt(DO) = 0.25 in the DO .mat files. Sampling internval 4 makes effective dt=1, which is also used in DP
Sampling_interval = 4

# multiplication factor: t > g.ni * c_ni
c_ni = 1.5

# multiplication factor for sin(theta) in r2 in const_rew_dt in custom_funcions.py
c_r2 = 1

#threshold
threshold = 1e-3