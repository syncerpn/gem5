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

_DATA_TIMESTAMP = 0
_DATA_PKGID = 1
_DATA_SRCROUTER = 2
_DATA_DSTROUTER = 3
_DATA_VNET = 4

_NODE_TO_NODE_DIST = 1 #cycle

class Progression:
    def __init__(self):
        self.schedule_list = {}
        self.last_key = None
        
    def add_schedule(self, schedule):
        if schedule.start != self.last_key:
            self.schedule_list[schedule.start]  = schedule.duration
            self.last_key = schedule.start
        else:
            if schedule.duration > self.schedule_list[schedule.start]:
                self.schedule_list[schedule.start] = schedule.duration
    
    def last_schedule(self):
        last_start_time = max(self.schedule_list.keys())
        return Schedule(last_start_time, self.schedule_list[last_start_time])
    
    def get_joined_segment(self):
        segment = []
        schedule_start_list = self.schedule_list.keys()
        for schedule_start in schedule_start_list:
            schedule_duration = self.schedule_list[schedule_start]
            if not segment:
                segment.append((schedule_start, schedule_start + schedule_duration))
                continue
            
            last_segment_start = segment[-1][0]
            last_segment_end = segment[-1][1]
            
            if schedule_start <= last_segment_end:
                segment[-1] = (last_segment_start, max(last_segment_end, schedule_start + schedule_duration))
            else:
                segment.append((schedule_start, schedule_start + schedule_duration))
        
        return segment

class Schedule:
    def __init__(self, start=0, duration=0):
        self.start = start
        self.duration = duration
    
    def __repr__(self):
        return f'[INFO] schedule @{self.start} finishes after {self.duration}'

class ScheduleData(Schedule):
    def __init__(self, data, row, col):
        self.start = data[_DATA_TIMESTAMP]
        src_id = get_mesh_index(data[_DATA_SRCROUTER], row, col)
        dst_id = get_mesh_index(data[_DATA_DSTROUTER], row, col)
        d = distance(src_id, dst_id)
        self.duration = 1 + d * _NODE_TO_NODE_DIST

def get_mesh_index(i, row, col):
    return (i % col, i // col)

def distance(a, b):
    xa, ya = a
    xb, yb = b
    return abs(xa - xb) + abs(ya - yb)

class Executor():
    def __init__(self, gem5_bin, sim_sys, num_cpus, workload_list, workload_args_list, protocol, mesh_row, mesh_col, n_port_per_node):
        self.trial_counter=0
        self.gem5_bin=gem5_bin
        self.sim_sys=sim_sys
        self.num_cpus=num_cpus
        self.workload_list=workload_list
        self.workload_args_list=workload_args_list
        self.protocol=protocol
        self.mesh_row=mesh_row
        self.mesh_col=mesh_col
        self.n_port_per_node=n_port_per_node

        self.last_trace_file = None #new: tracking produced trace file
        self.last_trace_config = None

        self.last_trace_content = [] #buffer this to speed up the whole process

        #fixed things
        self.patterns = ['simSeconds']
        self.trial_out_dir = '/hdd/nghiant/gem5_output_temp/freqminex_x86/'
        os.makedirs(self.trial_out_dir, exist_ok=True)

    def execute(self, config, output_trace=False):
        sim_secs = np.zeros(len(self.workload_list))

        trial_out_dir_counter = self.trial_out_dir + str(self.trial_counter) + '/'
        os.makedirs(trial_out_dir_counter, exist_ok=True)
        for wi, workload, workload_args in zip(list(range(len(self.workload_list))), self.workload_list, self.workload_args_list):
            print('[INFO] workload: ' + workload + ' ' + workload_args)
            trial_out_dir_counter_wi = trial_out_dir_counter + str(wi) + '/'
            os.makedirs(trial_out_dir_counter_wi, exist_ok=True)

            script_file = trial_out_dir_counter_wi + 'script'
            stats_file  = trial_out_dir_counter_wi + 'stats'
            placement_file = trial_out_dir_counter_wi + 'placement'

            simulation_cmd  = [self.gem5_bin]

            if output_trace:
                trace_file = trial_out_dir_counter_wi + 'trace'
                self.last_trace_file = trace_file
                self.last_trace_config = [i for i in config] #copy it for safety maybe
                self.last_trace_content = []
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
            simulation_cmd.append('--arm-iset=aarch64')
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
                print('unknown protocol')
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

        self.trial_counter += 1
        return sim_secs

    def estimate_using_last_trace(self, new_config):
        st = time.time()
        trace_file = self.last_trace_file
        config = self.last_trace_config

        # if self.protocol == 'moesi':
        #     device_name = ['L1Cache_Controller'] * self.num_cpus + ['L2Cache_Controller'] + ['Directory_Controller']
        #     device_id   = list(range(self.num_cpus)) + [0] + [0]

        if not self.last_trace_content:
            # id_dev = dict()
            # for device_name_i, device_id_i, port_i in zip(device_name, device_id, config):
            #     machine_name = device_name_i + '_' + str(device_id_i)
            #     id_dev[port_i] = machine_name

            print('[INFO] analyze and buffer new trace')
            with open(trace_file, 'r') as f:
                content = f.readlines()
            content = [line.strip('\n') for line in content]

            for data_line in content:
                items = data_line.split(' ')
                items = [item.strip(': []') for item in items if item.strip(': []')]
                items = items[2:] #remove the first two unused info
                
                data = [None, None, None, None, None]
                data[_DATA_TIMESTAMP] = int(items[0]) // 500
                data[_DATA_PKGID] = int(items[1])
                data[_DATA_SRCROUTER] = int(items[2])
                data[_DATA_DSTROUTER] = int(items[3])
                data[_DATA_VNET] = int(items[4])
                self.last_trace_content.append(data)

        new_dev_id = dict()
        for i, port_i in enumerate(new_config):
            new_dev_id[i] = port_i

        workload_progression_data = Progression()
        for data_raw in self.last_trace_content:
            data = [d for d in data_raw]
            data[_DATA_SRCROUTER] = new_dev_id[data[_DATA_SRCROUTER]]
            data[_DATA_DSTROUTER] = new_dev_id[data[_DATA_DSTROUTER]]
            
            workload_progression_data.add_schedule(ScheduleData(data, self.mesh_row, self.mesh_col)) 
        
        all_segment = workload_progression_data.get_joined_segment()
        data_transfer_cost = sum([s[1] - s[0] for s in all_segment]) * 500 / 1_000_000
        print('[INFO] estimate took: %.3f (s)' % (time.time() - st))
        return data_transfer_cost