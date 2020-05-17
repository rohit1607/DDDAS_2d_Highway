from utils.build_model_GPU import build_sparse_transition_model
import time

t1 = time.time()
build_sparse_transition_model(filename = 'GPU_Highway_', n_actions = 16, nt = 1)
t2 = time.time()
print("Time for 1 timestep = ", t2 - t1, " secs")