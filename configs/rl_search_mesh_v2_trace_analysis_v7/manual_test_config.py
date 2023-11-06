import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import os

import gem5_mesh_buffer_env as gem5gym
from executor import Executor

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
args = parser.parse_args()

#EXECUTOR
GEM5_BIN      = './build/X86_MOESI_CMP_directory/gem5.opt'
SIM_SYS       = './configs/simulation/sim_system.py'
OUT_DIR       = '/home/nghiant/git/gem5/manual_test/'

if not os.path.exists(OUT_DIR):
    os.system('mkdir ' + OUT_DIR)

PROTOCOL = 'moesi'
NUM_CPUS = 16 #32
MESH_COL = 8
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


TO_DEFINE_CONFIG = [0,8,16,24,2,10,18,26,4,12,20,28,6,14,22,30,1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31]

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

def main(config):
    # first run, need to get and analyze trace file
    perf_actual = env.check_performance(config, output_trace=True)
    perf = env.estimate_performance(config) #return estimated perf

if __name__ == '__main__':
    config = np.array(TO_DEFINE_CONFIG).astype(np.int32)
    main(config)