import numpy as np
import scipy.io
from utils.custom_functions import Map_vector_to_grid, interpolate_to_XP_grid
import matplotlib.pyplot as plt


#Read data from files
path_mat = scipy.io.loadmat('pathStore.mat')
grid_mat = scipy.io.loadmat('param.mat')
vel_mat = scipy.io.loadmat('000000.mat')
node_u = grid_mat['Nodeu']
node_v = grid_mat['Nodev']
XU = grid_mat['XU']
YU = grid_mat['YU']
XV = grid_mat['XV']
YV = grid_mat['YV']
XP =grid_mat['XP']
YP = grid_mat['YP']

u_mode = vel_mat['ui']
u_mean = vel_mat['u']
v_mode = vel_mat['vi']
v_mean = vel_mat['v']
Yi = vel_mat['Yi']

num_of_rzns = 5000

Vx_rzns = np.zeros((num_of_rzns,100,100))
Vy_rzns = np.zeros((num_of_rzns,100,100 ))


for k in range(num_of_rzns):
    u_rzn = u_mean + Yi[k] *u_mode
    Vx_uninterp = Map_vector_to_grid(u_rzn, node_u)
    Vx_rzns[k,:,:] = interpolate_to_XP_grid(XU, YU, Vx_uninterp, XP, YP)  # velocity field interpolated to grid

    v_rzn = v_mean + Yi[k] * v_mode
    Vy_uninterp = Map_vector_to_grid(v_rzn, node_v)
    Vy_rzns[k,:,:] = interpolate_to_XP_grid(XV, YV, Vy_uninterp, XP, YP)  # velocity field interpolated to grid


# Velx_rlzns = TemporaryFile()
np.save('Velx_5K_rlzns.npy', Vx_rzns)
np.save('Vely_5K_rlzns.npy', Vy_rzns)


plt.quiver(XP ,YP ,Vx_rzns[0,:,:] ,Vy_rzns[0,:,:])
plt.show()


"""
# for testing np.save and np.load
test = np.load('Velx_rlzns.npy')
for k in range(num_of_rzns):
    for i in range(100):
        for j in range(100):
            if Vx_rzns[k,i,j]!=test[k,i,j]:
                print(k,i,j,Vx_rzns[k,i,j],test[k,i,j])
print(test==Vx_rzns)
"""