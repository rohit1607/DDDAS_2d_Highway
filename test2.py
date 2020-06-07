# from utils.plot_functions import plot_paths_colored_by_EAT
#
# plotfile = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/7/QL/dt_size_5000/ALPHA_0.5/eps_0_0.25/Trajectories_after_exp'
# save_path = '/Users/rohit/workspace/ROHIT/DDDAS_2d_Highway/Experiments/7/QL/dt_size_5000/ALPHA_0.5/eps_0_0.25/'
# save_fname = 'colored_EAT'
# full_name = save_path + save_fname
# time_list = plot_paths_colored_by_EAT(plotFile = plotfile, savePath_fname=full_name)

# from utils.build_model_GPU import get_S_from_S_id

# gsize = 100
# print(get_S_from_S_id(2e4 + 3e2 + 47, gsize))
# print(get_S_from_S_id(2e4, gsize))
# print(get_S_from_S_id(3e2 + 47, gsize))
# print(get_S_from_S_id(47, gsize))

# from utils.setup_grid import setup_grid
# Nt = 40
# g, xs, ys, X, Y, Vx_rzns, Vy_rzns, num_rzns, path_mat, setup_params, setup_param_str = setup_grid(num_actions = 16, nt = Nt)

# ac_states = g.ac_state_space()

# for s in ac_states:
#     if s[0] == Nt -2:
#         print(s)


""" build_Vel_GPU  for  double gyre """
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from math import pi, sin, cos, atan2
import time
from collections import Counter
from multiprocessing import Pool
import pickle
from utils.setup_grid import setup_grid
# from utils.build_model import write_files
from definition import ROOT_DIR
from os.path import join, exists
from utils.custom_functions import createFolder, append_summary_to_summaryFile, read_pickled_File
from os import getcwd, makedirs


def extract_velocity(vel_field_data, t, i, j, rzn):
    # all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field_data
    #    0          1           2           3           4
    # vx = all_u_mat[t,i,j] + np.matmul(all_ui_mat[t, :, i,j], all_Yi[t, rzn,:])
    # vy = all_v_mat[t,i,j] + np.matmul(all_vi_mat[t, :, i,j], all_Yi[t, rzn,:])
    vx = vel_field_data[0][t, i, j] + np.matmul(vel_field_data[2][t, :, i, j],vel_field_data[4][t, rzn,:])
    vy = vel_field_data[1][t, i, j] + np.matmul(vel_field_data[3][t, :, i, j], vel_field_data[4][t, rzn,:])

    return vx,vy


def build_Vel_from_DOdata_GPU(vel_field_data, T):
    all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field_data
    nt, nrzns, nmodes = all_Yi.shape
    gsize = 100
    bDimx = 1000
    
    all_u_mat = all_u_mat.astype(np.float32)
    all_v_mat = all_v_mat.astype(np.float32)
    all_ui_mat = all_ui_mat.astype(np.float32)
    all_vi_mat = all_vi_mat.astype(np.float32)
    all_Yi = all_Yi.astype(np.float32)
    params = np.array([nt, nrzns, nmodes, gsize], dtype = np.float32)

    #vxrzns will not be there in final code. This is just for checking extracted velocities
    vxrzns_dummy = np.zeros((nrzns, gsize, gsize), dtype = np.float32)
    vyrzns_dummy = np.zeros((nrzns, gsize, gsize), dtype = np.float32)
    Tdummy = np.array([T,T]).astype(np.float32)


    all_u_mat_gpu = cuda.mem_alloc(all_u_mat.nbytes)
    all_v_mat_gpu = cuda.mem_alloc(all_v_mat.nbytes)
    all_ui_mat_gpu = cuda.mem_alloc(all_ui_mat.nbytes)
    all_vi_mat_gpu = cuda.mem_alloc(all_vi_mat.nbytes)
    all_Yi_gpu = cuda.mem_alloc(all_Yi.nbytes)
    params_gpu = cuda.mem_alloc(params.nbytes)
    T_gpu = cuda.mem_alloc(Tdummy.nbytes)


    vxrzns_gpu = cuda.mem_alloc(vxrzns_dummy.nbytes)
    vyrzns_gpu = cuda.mem_alloc(vyrzns_dummy.nbytes)


    cuda.memcpy_htod(all_u_mat_gpu, all_u_mat)
    cuda.memcpy_htod(all_v_mat_gpu, all_v_mat)
    cuda.memcpy_htod(all_ui_mat_gpu, all_ui_mat)
    cuda.memcpy_htod(all_vi_mat_gpu, all_vi_mat)
    cuda.memcpy_htod(all_Yi_gpu, all_Yi)
    cuda.memcpy_htod(params_gpu, params)
    cuda.memcpy_htod(T_gpu, Tdummy)

    cuda.memcpy_htod(vxrzns_gpu, vxrzns_dummy)
    cuda.memcpy_htod(vyrzns_gpu, vyrzns_dummy)


    mod = SourceModule("""

    __device__ int32_t get_thread_idx()
            // assigns idx to thread with which it accesses the flattened 3d vxrzns matrix
            // for a given T and a given action. 
            // runs for both 2d and 3d grid
            // TODO: may have to change this considering cache locality
        {
            // here i, j, k refer to a general matrix M[i][j][k]
            int32_t i = threadIdx.x;
            int32_t j = blockIdx.y;
            int32_t k = blockIdx.x;
            int32_t idx = k + (j*gridDim.x)  + (i*gridDim.x*gridDim.y)+ blockIdx.z*blockDim.x*gridDim.x*gridDim.y;
            return idx;
        }

    __device__ int32_t state1D_from_thread(int32_t T)
    {   
        // j ~ blockIdx.x
        // i ~ blockIdx.y 
        // The above three consitute a spatial state index from i and j of grid
        // last term is for including time index as well.
        return (blockIdx.x + (blockIdx.y*gridDim.x) + (T*gridDim.x*gridDim.y) ); 
    }

        __global__ void build_Vel_from_DOdata(float* T_arr, float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi, float* vxrzns, float* vyrzns, float* params)
        {
            int32_t nrzns = int32_t(params[1]);
            int32_t nmodes = int32_t(params[2]);
            int32_t T = T_arr[0];

            int32_t sp_uvi, str_uvi, sp_Yi, str_Yi; //startpoints and strides for accessing all_ui_mat, all_vi_mat and all_Yi
            float sum_x = 0;
            float sum_y = 0;
            float vx, vy, vx_mean, vy_mean;

            //thread index. also used to access resultant vxrzns[nrzns, gsize, gsize]
            int32_t idx = get_thread_idx();

            //rzn index to identify which of the 5k rzn it is. used to access all_Yi.
            int32_t rzn_id = (blockIdx.z * blockDim.x)  + threadIdx.x ;

            //mean_id is the index used to access the flattened all_u_mat[t,i,j].
            int32_t mean_id = state1D_from_thread(T);

            //to access all_ui_mat and all_vi_mat
            str_uvi = gridDim.x * gridDim.y;
            sp_uvi = (T * nmodes * str_uvi) + (gridDim.x * blockIdx.y) + (blockIdx.x);

            // to access all_Yi
            sp_Yi = (T * nrzns * nmodes) + (rzn_id * nmodes);

            if(idx < gridDim.x*gridDim.y*nrzns)
            {
                for(int i = 0; i < nmodes; i++)
                {
                    sum_x += all_ui_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
                }
                
                vx_mean = all_u_mat[mean_id];
                vx = vx_mean + sum_x;

                vxrzns[idx] = vx;
            }

            return;
        }


    """
    )


    func = mod.get_function("build_Vel_from_DOdata")
    func(T_gpu, all_u_mat_gpu, all_v_mat_gpu, all_ui_mat_gpu, all_vi_mat_gpu, all_Yi_gpu, vxrzns_gpu, vyrzns_gpu, params_gpu,
                    block=(bDimx, 1, 1), grid=(gsize, gsize, (nrzns // bDimx) + 1))
    
    vxrzns = np.empty_like(vxrzns_dummy).astype(np.float32)
    cuda.memcpy_dtoh(vxrzns, vxrzns_gpu)

    return vxrzns


def test_build():

    # TODO: Change data path
    data_path = '/home/rohit/Documents/Research/ICRA_2020/DG2/DG_Data/'
    all_u_mat = np.load(data_path +'all_u_mat.npy')
    all_ui_mat = np.load(data_path +'all_ui_mat.npy')
    all_v_mat = np.load(data_path +'all_v_mat.npy' )
    all_vi_mat = np.load(data_path +'all_vi_mat.npy')
    all_Yi = np.load(data_path +'all_Yi.npy' )
    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]

    T = 0
    nrzns = 5000
    gsize = 100
    cpu_vxrzns = np.zeros((nrzns, gsize, gsize), dtype = np.float32)
    for r in range(nrzns):
        for i in range(gsize):
            for j in range(gsize):
                cpu_vxrzns[r,i,j], _ = extract_velocity(vel_field_data, T, i, j, r)

    
    gpu_vxrzns = build_Vel_from_DOdata_GPU(vel_field_data, T)
    tnum = 1
    np.save('DG_vxrzns_cpu_' + str(tnum), cpu_vxrzns )
    np.save('DG_vxrzns_gpu_' + str(tnum), gpu_vxrzns )
    print("Running")


def test_compare(fcpu, fgpu):
    data_path = '/home/rohit/Documents/Research/ICRA_2020/DDDAS_2D_Highway/'
    cpu = np.load(data_path + fcpu)
    gpu = np.load(data_path + fgpu)
    count = 0

    for r in range(5000):
        for i in range(100):
            for j in range(100):
                if np.abs( cpu[r,i,j] - gpu[r,i,j]) > 1e-4 :
                    # print(r,i,j,'    ', cpu[r,i,j], '    ', gpu[r, i ,j])
                    count += 1
    print("badcount = ", count)



def test_print(fcpu, fgpu, R, I, J):
    data_path = '/home/rohit/Documents/Research/ICRA_2020/DDDAS_2D_Highway/'
    cpu = np.load(data_path + fcpu)
    gpu = np.load(data_path + fgpu)
    
    for r in range(R, R+2):
        for i in range(I,I +2):
            for j in range(J, J+2):
                print(r,i,j, '  ',cpu[r,i,j], ' ', gpu[r,i,j])
    

# test_compare('DG_vxrzns_cpu_1.npy', 'DG_vxrzns_gpu_1.npy')
test_print('DG_vxrzns_cpu_1.npy', 'DG_vxrzns_gpu_1.npy', 0, 50, 50)