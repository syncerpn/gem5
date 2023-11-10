import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os

import gem5_mesh_buffer_env as gem5gym
from executor import Executor
import model

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--save_dir', default='large_scale_SimpleFC1_train_min_improve_mesh_trace_analysis_v1_2_freqmine/', help='model configuration')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.system('mkdir ' + args.save_dir)

#EXECUTOR
GEM5_BIN      = './build/X86_MOESI_CMP_directory/gem5.opt'
SIM_SYS       = './configs/simulation/sim_system.py'
OUT_DIR       = '/home/nghiant/git/gem5/sim_output/'

if not os.path.exists(OUT_DIR):
    os.system('mkdir ' + OUT_DIR)

PROTOCOL = 'moesi'
NUM_CPUS = 8 #32
MESH_COL = 4
MESH_ROW = 4 #8
N_PORT_PER_NODE = 1
N_DEVICE = NUM_CPUS
if PROTOCOL == "moesi":
    N_DEVICE += (MESH_COL * MESH_ROW - NUM_CPUS) // 2 * 2
N_PORT = MESH_COL * MESH_ROW * N_PORT_PER_NODE

WORKLOAD_LIST      = [
                        '/home/nghiant/git/gem5/workload/darknet/darknet_x86',
                        # '/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/bodytrack/inst/amd64-linux.gcc-openmp/bin/bodytrack',
                        # '/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.gcc-openmp/bin/blackscholes',
                        # '/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/freqmine/inst/amd64-linux.gcc-openmp/bin/freqmine',
                      ]

WORKLOAD_ARGS_LIST = [
                        'kernel',
                        # 'kernel_medium',
                        # 'kernel_large',
                        # f'input_bodytrack_simdev/sequenceB_1 4 1 100 3 0 {NUM_CPUS}',
                        # 'input_freqmine_simdev/T10I4D100K_1k.dat 3',
                        # f'{NUM_CPUS} ../gem5/input_blackscholes_simsmall/in_4K.txt ../gem5/input_blackscholes_simsmall/test_output.txt',
                      ]


#RL SEARCH MODEL
POLICY_CONFIG = 'SimpleFC1'
STATE_DIM = N_DEVICE * N_PORT
EXPLORATION_FACTOR = 1
IMPROVEMENT_FACTOR = 10
LOG_INTERVAL = 100
BUFFER_DUMPING_INTERVAL = 1

SYNTHETIC_TRACE_FILE = "synthetic_trace_8"

# ENV
BUFFER_DIR = '_'.join(['buffer_save', PROTOCOL] + list(map(str, [NUM_CPUS, MESH_COL, MESH_ROW])))
if not os.path.exists(BUFFER_DIR):
    os.system('mkdir ' + BUFFER_DIR)

#============================================================================================
exe = Executor(gem5_bin=GEM5_BIN,
    sim_sys=SIM_SYS,
    num_cpus=NUM_CPUS,
    workload_list=WORKLOAD_LIST,
    workload_args_list=WORKLOAD_ARGS_LIST,
    protocol=PROTOCOL,
    mesh_row=MESH_ROW,
    mesh_col=MESH_COL,
    n_port_per_node=N_PORT_PER_NODE,
    out_dir=OUT_DIR
    )

env = gem5gym.gem5_mesh_buffer_env(
    n_device=N_DEVICE,
    n_port=N_PORT,
    buffer_dir=BUFFER_DIR,
    executer=exe,
    buffer_dumping_interval=BUFFER_DUMPING_INTERVAL
    )

policy_model = model.config_policy_model(POLICY_CONFIG, STATE_DIM, N_DEVICE, N_PORT)
optimizer = optim.Adam(policy_model.parameters(), lr=1e-2, weight_decay=0.0001)

PRESET_CONFIG = [0,4,8,12,3,7,11,15,5,9,6,10,1,13,2,14]
PRESET_CONFIG = [9,8,1,6,4,2,3,12,10,5,14,0,7,11,15,13]

def main():
    # first run, need to get and analyze trace file
    exe.last_trace_file = SYNTHETIC_TRACE_FILE
    config = np.array(PRESET_CONFIG).astype(np.int32)
    perf = env.estimate_performance(config) #return estimated perf
    print(f"[INFO] init config ({perf:.3f}): {config}")

if __name__ == '__main__':
    main()