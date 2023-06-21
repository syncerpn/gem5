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
from executor import SystemExecutor, NetworkExecutor
import model

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
args = parser.parse_args()

#NORMAL SYSTEM
SYS_BIN = './build/X86_MOESI_CMP_directory/gem5.opt'
SYS_DEF = './configs/simulation/sim_system.py'
SYS_OUT = './sim_output/'

#NETWORK TRACE SIm
NET_BIN = './build/Garnet_trace/gem5.opt'
NET_DEF = './configs/simulation/sim_network.py'
NET_OUT = SYS_OUT + 'net_output/'

POLICY_MODEL_SAVE_DIR = SYS_OUT + 'model/'

if not os.path.exists(SYS_OUT):
    os.system('mkdir ' + SYS_OUT)

if not os.path.exists(NET_OUT):
    os.system('mkdir ' + NET_OUT)

if not os.path.exists(POLICY_MODEL_SAVE_DIR):
    os.system('mkdir ' + POLICY_MODEL_SAVE_DIR)

WORKLOAD_LIST      = [
                        # '/home/nghiant/git/gem5/workload/darknet/darknet_arm',
                        # '/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/bodytrack/inst/amd64-linux.gcc-openmp/bin/bodytrack',
                        '/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.gcc-openmp/bin/blackscholes',
                        # '/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/freqmine/inst/amd64-linux.gcc-openmp/bin/freqmine',
                      ]

WORKLOAD_ARGS_LIST = [
                        # 'kernel',
                        # 'input_bodytrack_simdev/sequenceB_1 4 1 100 3 0 32',
                        # '32 ../gem5/input_blackscholes_simsmall/in_4K.txt ../gem5/input_blackscholes_simsmall/test_output.txt',
                        '32 ../gem5/input_blackscholes_simdev/in_16.txt ../gem5/input_blackscholes_simdev/test_output.txt',
                        # 'input_freqmine_simdev/T10I4D100K_1k.dat 3',
                      ]

PROTOCOL = 'moesi'
NUM_CPUS = 32
MESH_COL = 8
MESH_ROW = 8
N_PORT_PER_NODE = 1
N_DEVICE = NUM_CPUS + 2
N_PORT = MESH_COL * MESH_ROW * N_PORT_PER_NODE

#RL SEARCH MODEL
POLICY_CONFIG = 'SimpleFC1'
STATE_DIM = N_DEVICE * N_PORT
EXPLORATION_FACTOR = 1
IMPROVEMENT_FACTOR = 10
LOG_INTERVAL = 100
BUFFER_DUMPING_INTERVAL = 1

# ENV
BUFFER_DIR = '_'.join(['buffer_save', PROTOCOL] + list(map(str, [NUM_CPUS, MESH_COL, MESH_ROW])))
if not os.path.exists(BUFFER_DIR):
    os.system('mkdir ' + BUFFER_DIR)

#============================================================================================
sys_exe = SystemExecutor(
    gem5_bin=SYS_BIN,
    sim_sys=SYS_DEF,
    num_cpus=NUM_CPUS,
    workload_list=WORKLOAD_LIST,
    workload_args_list=WORKLOAD_ARGS_LIST,
    protocol=PROTOCOL,
    mesh_row=MESH_ROW,
    mesh_col=MESH_COL,
    n_port_per_node=N_PORT_PER_NODE,
    )

net_exe = NetworkExecutor(
    gem5_bin=NET_BIN,
    sim_sys=NET_DEF,
    num_cpus=NUM_CPUS,
    n_workload=len(WORKLOAD_LIST),
    protocol=PROTOCOL,
    mesh_row=MESH_ROW,
    mesh_col=MESH_COL,
    n_port_per_node=N_PORT_PER_NODE,
    )

env = gem5gym.gem5_mesh_buffer_env(
    n_device=N_DEVICE,
    n_port=N_PORT,
    buffer_dir=BUFFER_DIR,
    executer=sys_exe,
    estimator=net_exe,
    buffer_dumping_interval=BUFFER_DUMPING_INTERVAL)

policy_model = model.config_policy_model(POLICY_CONFIG, STATE_DIM, N_DEVICE, N_PORT)
optimizer = optim.Adam(policy_model.parameters(), lr=1e-2, weight_decay=0.0001)

def main():
    # first run, need to get and analyze trace file

    config, state = env.random_state()
    perf_actual = env.check_performance(config)
    perf = env.estimate_performance(config) #return estimated perf

    best_perf = perf
    best_config = config.copy()
    best_state = state.copy()

    # rl search start
    running_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        print('[INFO] episode', i_episode)

        EXPLORATION_FACTOR = 1
        IMPROVEMENT_FACTOR = 10

        # reset environment
        # print('[INFO] --initial phase')

        perf = best_perf
        config = best_config.copy()
        state = best_state.copy()


        action_d_prob, action_p_prob, state_value = policy_model(torch.from_numpy(state).float())

        # exploit the action
        action_d_prob_map   = action_d_prob.data.clone()
        action_p_prob_map   = action_p_prob.data.clone()
        action_d_map   = torch.argmax(action_d_prob_map, dim=-1)
        action_p_map   = torch.argmax(action_p_prob_map, dim=-1)

        # print('[INFO] --exploit phase')
        new_config_map, new_state_map = env.step(config, action_d_map.numpy(), action_p_map.numpy())
        new_perf_map = self.estimate_performance(new_config_map)

        if new_perf_map < best_perf:
            best_perf = new_perf_map
            best_config = new_config_map.copy()
            best_state = new_state_map.copy()

        # exploit the action

        # explore the action
        action_d_prob = (action_d_prob * 0.7 + (1 - action_d_prob) * 0.3)
        action_d_prob /= torch.sum(action_d_prob)
        action_p_prob = (action_p_prob * 0.7 + (1 - action_p_prob) * 0.3)
        action_p_prob /= torch.sum(action_p_prob)

        m_d = Categorical(action_d_prob)
        m_p = Categorical(action_p_prob)

        # and sample an action using the distribution
        action_d = m_d.sample()
        action_p = m_p.sample()

        # take the action
        # print('[INFO] --explore phase')
        new_config, new_state, new_perf = env.step(config, action_d.numpy(), action_p.numpy())
        new_perf = self.estimate_performance(new_config)

        if new_perf < best_perf:
            best_perf = new_perf
            best_config = new_config.copy()
            best_state = new_state.copy()


        reward = -(new_perf - perf).mean()
        running_reward = 0.1 * reward + 0.9 * running_reward

        # perform backprop
        advantage = -(new_perf - new_perf_map) * EXPLORATION_FACTOR + -(new_perf - perf) * IMPROVEMENT_FACTOR

        policy_loss = -(m_d.log_prob(action_d) + m_p.log_prob(action_p)) * torch.Tensor(np.array(advantage))
        value_loss = F.smooth_l1_loss(state_value, torch.tensor(new_perf_map).float())

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = policy_loss.sum() + value_loss.sum()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        # log results
        if i_episode % LOG_INTERVAL == 0:
            print('[INFO] Episode {}\tLast reward: {:.5f}\tAverage reward: {:.5f}\tLoss: {:.5f}'.format(
                  i_episode, reward, running_reward, loss))
            policy_model.save(args.save_dir, i_episode)
            print('[INFO] best estimated performance %f' % best_perf)
            print('[INFO] best estimated config ', best_config)
            perf_actual = env.check_performance(best_config)
            env.dump_buffer_to_file()

if __name__ == '__main__':
    main()