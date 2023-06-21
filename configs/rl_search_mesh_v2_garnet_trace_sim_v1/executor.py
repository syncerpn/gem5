import os
import re
import optparse
import numpy as np
import signal
import subprocess
import itertools
import time

def parse_output(file, patterns):
    with open(file) as f:
        data = f.readlines()

    data = [line.strip('\n') for line in data]

    result = []
    for p in patterns:
        result += [line for line in data if re.search(p, line)]

    return result

def get_mesh_index(i, row, col):
    return (i % col, i // col)

class SystemExecutor():
    def __init__(self, gem5_bin, sim_sys, num_cpus, workload_list, workload_args_list, protocol, mesh_row, mesh_col, n_port_per_node):
        self.gem5_bin=gem5_bin
        self.sim_sys=sim_sys
        self.num_cpus=num_cpus
        self.workload_list=workload_list
        self.workload_args_list=workload_args_list
        self.protocol=protocol
        self.mesh_row=mesh_row
        self.mesh_col=mesh_col
        self.n_port_per_node=n_port_per_node

        self.trace = None #new: tracking produced trace file

        #fixed things
        self.patterns = ['simSeconds']
        self.trial_out_dir = '/hdd/nghiant/gem5_output_temp/_run_tmp_sys/'
        os.makedirs(self.trial_out_dir, exist_ok=True)

    def folder_name_from_config(self, config):
        return self.trial_out_dir + '_'.join([str(x) for x in config]) + '/'

    def execute(self, config):
        trial_out_dir_counter = self.folder_name_from_config(config)
        os.makedirs(trial_out_dir_counter, exist_ok=True)

        for wi, workload, workload_args in zip(list(range(len(self.workload_list))), self.workload_list, self.workload_args_list):
            print('[INFO] workload: ' + workload + ' ' + workload_args)
            trial_out_dir_counter_wi = trial_out_dir_counter + str(wi) + '/'
            os.makedirs(trial_out_dir_counter_wi, exist_ok=True)

            script_file = trial_out_dir_counter_wi + 'script'
            stats_file  = trial_out_dir_counter_wi + 'stats'
            placement_file = trial_out_dir_counter_wi + 'placement'

            simulation_cmd  = [self.gem5_bin]

            #will feed this to network executor
            trace_file = trial_out_dir_counter_wi + 'trace'
            self.trace = trace_file

            simulation_cmd.append('--debug-flags=nghiant_RubyNetwork')
            simulation_cmd.append('--debug-file=' + trace_file)

            simulation_cmd.append('--outdir=' + trial_out_dir_counter_wi)
            simulation_cmd.append('--stats-file=' + stats_file)
            simulation_cmd.append(self.sim_sys)
            simulation_cmd.append('--cmd=' + workload)
            simulation_cmd.append('--options=' + '"' + workload_args + '"')
            simulation_cmd.append('--ruby')
            simulation_cmd.append('--network=garnet')
            simulation_cmd.append('--topology=Mesh_nghiant_custom_v2')
            simulation_cmd.append('--placement-file=' + placement_file)
            simulation_cmd.append('--num-cpus=' + str(self.num_cpus))
            simulation_cmd.append('--cpu-type=DerivO3CPU')
            simulation_cmd.append('--cpu-clock=2GHz')
            simulation_cmd.append('--mem-type=DDR4_2400_16x4')
            simulation_cmd.append('--mem-size=4GB')
            simulation_cmd.append('--mem-channels=4')
            simulation_cmd.append('--mem-ranks=2')
            simulation_cmd.append('--l1i_size=64kB')
            simulation_cmd.append('--l1i_assoc=16')
            simulation_cmd.append('--l1d_size=64kB')
            simulation_cmd.append('--l1d_assoc=16')
            simulation_cmd.append('--l2_size=128MB')
            simulation_cmd.append('--l2_assoc=16')
            simulation_cmd.append('--cacheline_size=64')

            simulation_cmd = ' '.join(simulation_cmd)
            script_file_f = open(script_file, 'w')

            #update input mesh config
            if self.protocol == 'moesi':
                n_port = self.mesh_row * self.mesh_col * self.n_port_per_node
                device_name = ['L1Cache_Controller'] * self.num_cpus + ['L2Cache_Controller'] + ['Directory_Controller']
                device_id = list(range(self.num_cpus)) + [0] + [0]
                n_device = len(device_name)
                assert(n_device <= n_port)

            else:
                print('[ERRO] unknown protocol')
                assert(0)

            with open(placement_file, 'w') as f:
                lines = [str(self.mesh_row) + ' ' + str(self.mesh_col)]
                for i in range(n_device):
                    d_name = device_name[i]
                    d_id   = device_id[i]
                    d_port = config[i]
                    
                    d_node = d_port // self.n_port_per_node
                    d_node_col = d_node %  self.mesh_col
                    d_node_row = d_node // self.mesh_col

                    placement_line = [d_name, str(d_id), str(d_node_row), str(d_node_col)]
                    lines += [' '.join(placement_line)]
                
                f.write('\n'.join(lines))

            #run the new config
            p = subprocess.Popen(simulation_cmd, shell=True, stdout=script_file_f, stderr=subprocess.STDOUT).wait()

            #measure the performance
            result = parse_output(stats_file, self.patterns)
            result_list = result[0].split(' ')
            result_list = [t for t in result_list if t != '']
            sim_sec = float(result_list[1]) * 1000000.0 #(microsecs)

            sim_secs[wi] = sim_sec

        return sim_secs

class NetworkExecutor:
    def __init__(self, gem5_bin, sim_sys, num_cpus, n_workload, protocol, mesh_row, mesh_col, n_port_per_node):

        self.trial_counter=0
        self.gem5_bin=gem5_bin
        self.sim_sys=sim_sys
        self.num_cpus=num_cpus
        self.n_workload = n_workload
        self.protocol=protocol
        self.mesh_row=mesh_row
        self.mesh_col=mesh_col
        self.n_port_per_node=n_port_per_node

        self.trace = None

        #fixed things
        self.patterns = ['system.ruby.network.average_packet_latency']
        self.trial_out_dir = '/hdd/nghiant/gem5_output_temp/_run_tmp_net/'
        os.makedirs(self.trial_out_dir, exist_ok=True)

    def execute(self, config):
        perfs = np.zeros(self.n_workload)

        trial_out_dir_counter = self.trial_out_dir + str(self.trial_counter) + '/'
        os.makedirs(trial_out_dir_counter, exist_ok=True)
        for wi, workload, workload_args in zip(list(range(len(self.workload_list))), self.workload_list, self.workload_args_list):
            print('[INFO] workload: ' + workload + ' ' + workload_args)
            trial_out_dir_counter_wi = trial_out_dir_counter + str(wi) + '/'
            os.makedirs(trial_out_dir_counter_wi, exist_ok=True)

            script_file = trial_out_dir_counter_wi + 'script'
            stats_file  = trial_out_dir_counter_wi + 'stats'
            placement_file = trial_out_dir_counter_wi + 'placement'
            
            converted_trace = self.trace + '_converted'

            simulation_cmd  = [self.gem5_bin]

            simulation_cmd.append('--outdir=' + trial_out_dir_counter_wi)
            simulation_cmd.append('--stats-file=' + stats_file)
            simulation_cmd.append(self.sim_sys)
            simulation_cmd.append('--ruby')
            simulation_cmd.append('--network=garnet')
            simulation_cmd.append('--topology=Mesh_gn')
            simulation_cmd.append('--placement-file=' + placement_file)
            simulation_cmd.append('--trace=' + converted_trace)
            simulation_cmd.append('--num-cpus=' + str(self.num_cpus))
            simulation_cmd.append('--cpu-clock=2GHz')
            simulation_cmd.append('--mem-type=DDR4_2400_16x4')
            simulation_cmd.append('--mem-size=4GB')
            simulation_cmd.append('--mem-channels=4')
            simulation_cmd.append('--mem-ranks=2')
            simulation_cmd.append('--l1i_size=64kB')
            simulation_cmd.append('--l1i_assoc=16')
            simulation_cmd.append('--l1d_size=64kB')
            simulation_cmd.append('--l1d_assoc=16')
            simulation_cmd.append('--l2_size=128MB')
            simulation_cmd.append('--l2_assoc=16')
            simulation_cmd.append('--cacheline_size=64')
            simulation_cmd.append('--sim-cycles=800000000')

            simulation_cmd = ' '.join(simulation_cmd)
            script_file_f = open(script_file, 'w')

            #update input mesh config
            if self.protocol == 'moesi':
                n_port = self.mesh_row * self.mesh_col * self.n_port_per_node
                device_name = ['L1Cache_Controller'] * self.num_cpus + ['L2Cache_Controller'] + ['Directory_Controller']
                device_id = list(range(self.num_cpus)) + [0] + [0]
                n_device = len(device_name)
                assert(n_device <= n_port)

            else:
                print('[ERRO] unknown protocol')
                assert(0)

            with open(placement_file, 'w') as f:
                lines = [str(self.mesh_row) + ' ' + str(self.mesh_col)]
                for i in range(n_device):
                    d_name = device_name[i]
                    d_id   = device_id[i]
                    d_port = config[i]
                    
                    d_node = d_port // self.n_port_per_node
                    d_node_col = d_node %  self.mesh_col
                    d_node_row = d_node // self.mesh_col

                    placement_line = [d_name, str(d_id), str(d_node_row), str(d_node_col)]
                    lines += [' '.join(placement_line)]
                
                f.write('\n'.join(lines))

            with open(self.trace, 'r') as f:
                with open(converted_trace, 'w') as nf:
                    data = f.readline().strip('\n')
                    while data:
                        cycle, s, d, v = map(int, data.split(' '))
                        new_d = occupied[d]
                        nf.write(' '.join([str(cycle),str(s),str(new_d),str(v)])+'\n')
                            
                        data = f.readline().strip('\n')

            #run the new config
            p = subprocess.Popen(simulation_cmd, shell=True, stdout=script_file_f, stderr=subprocess.STDOUT).wait()

            #measure the performance
            result = parse_output(stats_file, self.patterns)
            result_list = result[0].split(' ')
            result_list = [t for t in result_list if t != '']
            perf = float(result_list[1]) #(nanasec ??)

            perfs[wi] = perf

        self.trial_counter += 1
        return perfs