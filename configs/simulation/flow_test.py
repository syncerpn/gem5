import os
import re
import optparse
import numpy as np
import signal
import subprocess

parser = optparse.OptionParser()

parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/ARM_MOESI_CMP_directory_fast/gem5.fast", help="gem5 build")
parser.add_option("-s",  "--sim-sys",        type="string", default="/home/nghiant/git/gem5/configs/simulation/sim_system.py", help="simulation system")
parser.add_option("-n",  "--num-cpus",       type="int",    default=4, help="number of cpus")
parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/sim_output/", help="output dir")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/arm/blackscholes_openmp", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 workload/blackscholes_in_16", help="workload arguments")

parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/darknet/darknet_arm", help="workload")
parser.add_option("-a",  "--workload-args",  type="string", default="kernel", help="workload arguments")

parser.add_option("-p",  "--protocol",       type="string", default="moesi", help="protocol")
parser.add_option("-r",  "--mesh-row",       type="int",    default=2, help="mesh max row")
parser.add_option("-c",  "--mesh-col",       type="int",    default=4, help="mesh max column")

parser.add_option("-e",  "--epoch",          type="int",    default=30000, help="number of epochs/trials")

(options, args) = parser.parse_args()

def parse_output(file, patterns):
    with open(file) as f:
        data = f.readlines()

    data = [line.strip('\n') for line in data]

    result = []
    for p in patterns:
        result += [line for line in data if re.search(p, line)]

    return result

def random_device_placement(n_device, n_port):
    probs = np.abs(np.random.randn(n_device, n_port))

    occupied = []

    for d in range(n_device):
        probs_d = probs[d,:]
        for i in occupied: #remove occupied port
            probs_d[i] = 0.0
        probs_d /= np.sum(probs_d)
        occupied.append(np.argmax(probs_d))
    return occupied

import itertools
greedy_index = 0

def greedy_device_placement(n_device, n_port):
    global greedy_index
    all_permus = np.array(list(set(list(itertools.permutations(list(range(n_device)) + [-1]*(n_port-n_device))))))

    if (greedy_index == 0):
        print('[ INFO ] greedy placement has been used; possible: %10d placements' % len(all_permus))
    
    assert(greedy_index < len(all_permus)) #auto quit
    node_occupied = all_permus[greedy_index]
    occupied = [np.where(node_occupied == i)[0][0] for i in range(n_device)]
    greedy_index = greedy_index + 1
    return occupied

#hyper-param settings
n_port_per_node = 1
pattern = ['simSeconds']

#loop to death
while greedy_index < options.epoch:
    #prepare command
    trial_out_dir = options.out_dir + 'trial_' + str(greedy_index) + '/'
    
    os.makedirs(trial_out_dir, exist_ok=True)

    script_file = trial_out_dir + 'script'
    stats_file  = trial_out_dir + 'stats'
    placement_file = trial_out_dir + 'placement'

    simulation_cmd  = [options.gem5_bin]
    simulation_cmd.append('--outdir=' + trial_out_dir)
    simulation_cmd.append('--stats-file=' + stats_file)
    simulation_cmd.append(options.sim_sys)
    simulation_cmd.append('--cmd=' + options.workload)
    simulation_cmd.append('--options=' + '"' + options.workload_args + '"')
    simulation_cmd.append('--ruby')
    simulation_cmd.append('--network=garnet')
    simulation_cmd.append('--topology=Mesh_nghiant_custom')
    simulation_cmd.append('--placement-file=' + placement_file)
    simulation_cmd.append('--num-cpus=' + str(options.num_cpus))
    simulation_cmd.append('--cpu-type=O3_ARM_v7a_3')
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
    simulation_cmd.append('| tee ' + script_file)

    simulation_cmd = ' '.join(simulation_cmd)

    #update input mesh config
    if options.protocol == 'moesi':
        n_port = options.mesh_row * options.mesh_col * n_port_per_node
        device_name = ['L1Cache_Controller'] * options.num_cpus + ['L2Cache_Controller'] + ['Directory_Controller']
        device_id = list(range(options.num_cpus)) + [0] + [0]
        n_device = len(device_name)
        assert(n_device <= n_port)

        occupied = greedy_device_placement(n_device, n_port)

    else:
        print('unknown protocol')
        assert(0)

    with open(placement_file, 'w') as f:
        lines = []
        for i in range(n_device):
            d_name = device_name[i]
            d_id   = device_id[i]
            d_port = occupied[i]
            
            d_node = d_port // n_port_per_node
            d_node_col = d_node %  options.mesh_col
            d_node_row = d_node // options.mesh_col

            placement_line = [d_name, str(d_id), str(d_node_row), str(d_node_col)]
            lines += [' '.join(placement_line)]
        
        f.write('\n'.join(lines))

    #run the new config
    p = subprocess.Popen(simulation_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).wait()

    #measure the performance
    result = parse_output(stats_file,pattern)
    result_list = result[0].split(' ')
    result_list = [t for t in result_list if t != '']
    sim_sec = float(result_list[1])

    print('[ INFO ] trial %3d | binary execution time: %.6f' % (greedy_index,sim_sec))

    #update the estimation model/search algorithm