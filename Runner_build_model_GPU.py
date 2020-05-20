from utils.build_model_GPU import build_sparse_transition_model
from utils.build_model import Build_Model
import time

t1 = time.time()
# build_sparse_transition_model(filename = 'GPU_test_7_', n_actions = 16, nt = 3 )
build_sparse_transition_model(filename = 'GPU_testGrid_6_', n_actions = 16, Test_grid= True )

Build_Model(filename='CPU_testGrid_6_',n_actions=16, Test_grid= True )

t2 = time.time()