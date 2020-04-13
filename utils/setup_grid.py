from grid_world_stationary import timeOpt_grid
from utils.custom_functions import my_meshgrid
from definition import ROOT_DIR
import scipy.io
import numpy as np
from os import getcwd
from os.path import join
import math


def get_filler_coords(traj, start_pos):
    """
    generates points on a line joining start position and the first point on a given path
    :param traj: 1 path
    :param start_pos: start pos of problem
    :return: list of (x,y) coords
    """
    x0, y0 = start_pos
    x, y = traj[0]
    #num_points: no. of filler points is based on trajectory's waypoint density.
    num_points = int(np.linalg.norm(traj[0] - np.array([x0, y0]), 2) // np.linalg.norm(traj[0] - traj[1], 2))
    filler_xy = np.linspace((x0, y0), (x, y), int(num_points), endpoint=False)
    return filler_xy


def prune_and_pad_paths(path_ndarray, start_xy, end_xy):
    """
    prunes out waypoints that go past endpos. pads way points between startpos and trajectories' start
    :param path_ndarray: list/ndarray of paths
    :param start_xy: starpos
    :param end_xy: endpos
    :return: pruned and padded path_ndarray
    """
    xf, yf = end_xy
    _, num_rzns = path_ndarray.shape
    for n in range(num_rzns):
        # prune path
        l = len(path_ndarray[0, n])
        idx_list = []
        for i in range(l - 8, l):
            x, y = path_ndarray[0, n][i]
            if x < xf and y > yf:
                idx_list.append(i)
            elif math.isnan(x) or math.isnan(y):
                idx_list.append(i)
        path_ndarray[0, n] = np.delete(path_ndarray[0, n], idx_list, axis=0)

        # pad path
        filler = get_filler_coords(path_ndarray[0, n], start_xy)
        path_ndarray[0, n] = np.append(filler, path_ndarray[0, n], axis=0)
    return path_ndarray


#Tweaked
def setup_grid(num_actions =16, nt = 100, dt =1, F =1, startpos = (79, 49), endpos = (20, 50), Test_grid= False):
    """
    sets up discretized grid based on input data.
    prunes and pads paths (preprocesing)
    loads velocity data in line with discritized grid. velocities are interpolated as a pre-processing step.
    Contains two grid setups: 1. Highway problem    2. small self generated test problem
    :param num_actions:
    :param nt:
    :param dt:
    :param F:
    :param startpos:
    :param endpos:
    :param Test_grid:
    :return:
    """
    if Test_grid == False:
        #Read data from files
        grid_mat = scipy.io.loadmat(join(ROOT_DIR, 'Input_data_files/param.mat'))
        path_mat = scipy.io.loadmat(join(ROOT_DIR, 'Input_data_files/pathStore.mat'))
        paths = prune_and_pad_paths(path_mat['pathStore'], (49.5, 20.5), (50.5, 79.5))

        XP = grid_mat['XP']
        YP = grid_mat['YP']
        Vx_rzns = np.load(join(ROOT_DIR,'Input_data_files/Velx_5K_rlzns.npy'))
        Vy_rzns = np.load(join(ROOT_DIR,'Input_data_files/Vely_5K_rlzns.npy'))
        num_rzns = Vx_rzns.shape[0]
        param_str = ['num_actions', 'nt', 'dt', 'F', 'startpos', 'endpos']
        params = [num_actions, nt, dt, F, startpos, endpos]

        #Set up Grid
        xs = XP[1,:]
        ys_temp = YP[:,1]
        ys = np.flip(ys_temp)
        X, Y = my_meshgrid(xs, ys)
        vxs = [0.0, 0.5, 1]
        vys = [0.0, 0.5, 1]

    else:
        num_rzns = 50
        size = 12
        xs = np.arange(size)
        ys = np.arange(size)
        vxs = [0.0, 0.5, 1]
        vys = [0.0, 0.5, 1]
        X, Y = my_meshgrid(xs, ys)
        Vx_rzns = np.zeros((num_rzns, size, size))
        Vy_rzns = np.zeros((num_rzns, size, size))
        Vx_rzns[:,5, :] = 1


        paths = None
        startpos = (10,4)
        endpos  = (2,4)
        nt = 20
        dt = 1
        num_actions = 16
        F = 1

        param_str = ['num_actions', 'nt', 'dt', 'F', 'startpos', 'endpos']
        params = [num_actions, nt, dt, F, startpos, endpos]

    g = timeOpt_grid(xs, ys, vxs, vys, dt, nt, F, startpos, endpos, num_actions=num_actions)

    print("Grid Setup Complete !")
    # CHANGE RUNNER FILE TO GET PARAMS(9TH ARG) IF YOU CHANGE ORDER OF RETURNS HERE

    return g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, paths, params, param_str




"""
def setup_grid(num_actions =16, nt = 100, dt =1, F =1, startpos = (78, 48), endpos = (20, 50), Test_grid= False):

    if Test_grid == False:
        #Read data from files
        grid_mat = scipy.io.loadmat(join(ROOT_DIR, 'Input_data_files/param.mat'))
        path_mat = scipy.io.loadmat(join(ROOT_DIR, 'Input_data_files/pathStore.mat'))
        XP = grid_mat['XP']
        YP = grid_mat['YP']
        Vx_rzns = np.load(join(ROOT_DIR,'Input_data_files/Velx_5K_rlzns.npy'))
        Vy_rzns = np.load(join(ROOT_DIR,'Input_data_files/Vely_5K_rlzns.npy'))
        num_rzns = Vx_rzns.shape[0]
        param_str = ['num_actions', 'nt', 'dt', 'F', 'startpos', 'endpos']
        params = [num_actions, nt, dt, F, startpos, endpos]

        #Set up Grid
        xs = XP[1,:]
        ys_temp = YP[:,1]
        ys = np.flip(ys_temp)
        X, Y = my_meshgrid(xs, ys)

    else:
        num_rzns = 50
        size = 12
        xs = np.arange(size)
        ys = np.arange(size)
        X, Y = my_meshgrid(xs, ys)
        Vx_rzns = np.zeros((num_rzns, size, size))
        Vy_rzns = np.zeros((num_rzns, size, size))
        Vx_rzns[:,5, :] = 1


        path_mat = None
        startpos = (10,4)
        endpos  = (2,4)
        nt = 20
        dt = 1
        num_actions = 16
        F = 1

        param_str = ['num_actions', 'nt', 'dt', 'F', 'startpos', 'endpos']
        params = [num_actions, nt, dt, F, startpos, endpos]


    g = timeOpt_grid(xs, ys, dt, nt, F, startpos, endpos, num_actions=num_actions)

    print("Grid Setup Complete !")
    # CHANGE RUNNER FILE TO GET PARAMS(9TH ARG) IF YOU CHANGE ORDER OF RETURNS HERE

    return g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, params, param_str

"""