import os
import re
import optparse
import numpy as np
import signal
import subprocess
import itertools

# np.random.seed(1)

parser = optparse.OptionParser()

parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/ARM_MOESI_CMP_directory_fast/gem5.fast", help="gem5 build")
parser.add_option("-s",  "--sim-sys",        type="string", default="/home/nghiant/git/gem5/configs/simulation/sim_system.py", help="simulation system")
parser.add_option("-n",  "--num-cpus",       type="int",    default=32, help="number of cpus")
parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/sim_output_blackscholes32_random_mesh_cache_dependency/", help="output dir")

parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/arm/blackscholes_openmp", help="workload")
parser.add_option("-a",  "--workload-args",  type="string", default="32 workload/in_512.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/arm/blackscholes_openmp", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 workload/in_4.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/darknet/darknet_arm", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="kernel", help="workload arguments")

parser.add_option("-p",  "--protocol",       type="string", default="moesi", help="protocol")
parser.add_option("-r",  "--mesh-row",       type="int",    default=8, help="mesh max row")
parser.add_option("-c",  "--mesh-col",       type="int",    default=8, help="mesh max column")

(options, args) = parser.parse_args()

CACHE_LINE_BASE = 16
CACHE_L1_SIZE_BASE  = 16
CACHE_L1_ASSOC_BASE = 8
CACHE_L2_SIZE_BASE  = 16
CACHE_L2_ASSOC_BASE = 8

N_CACHE_L1_SIZE  = 4
N_CACHE_L2_SIZE  = 4

N_CACHE_OPT = N_CACHE_L1_SIZE * N_CACHE_L2_SIZE

def parse_output(file, patterns):
    with open(file) as f:
        data = f.readlines()

    data = [line.strip('\n') for line in data]

    result = []
    for p in patterns:
        result += [line for line in data if re.search(p, line)]

    return result

def random_device_placement(n_device, n_port):
    all_ports = list(range(n_port))
    np.random.shuffle(all_ports)
    occupied = all_ports[:n_device]
    return occupied

#hyper-param settings
n_port_per_node = 1
pattern = ['simSeconds']

#prepare command
best_perf = 99999
for ei in range(10):
    #update input mesh config

    trial_out_dir = options.out_dir + 'trial_random' + str(ei) + '/'

    os.makedirs(trial_out_dir, exist_ok=True)

    script_file = trial_out_dir + 'script'
    stats_file  = trial_out_dir + 'stats'
    placement_file = trial_out_dir + 'placement'

    if options.protocol == 'moesi':
        n_port = options.mesh_row * options.mesh_col * n_port_per_node
        device_name = ['L1Cache_Controller'] * options.num_cpus + ['L2Cache_Controller'] + ['Directory_Controller']
        device_id = list(range(options.num_cpus)) + [0] + [0]
        n_device = len(device_name)
        assert(n_device <= n_port)

        occupied = random_device_placement(n_device, n_port)
        print('[ INFO ] occupied', occupied)

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

    all_perfs = []
    best_perf = 99999
    best_ci = -1
    for ci in range(N_CACHE_OPT):
        c1s = ci  % N_CACHE_L1_SIZE
        c2s = ci // N_CACHE_L1_SIZE

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
        simulation_cmd.append('--l1i_size=%dkB' % (CACHE_L1_SIZE_BASE * (1 << c1s)))
        simulation_cmd.append('--l1i_assoc=16')
        simulation_cmd.append('--l1d_size=%dkB' % (CACHE_L1_SIZE_BASE * (1 << c1s)))
        simulation_cmd.append('--l1d_assoc=16')
        simulation_cmd.append('--l2_size=%dMB' % (CACHE_L2_SIZE_BASE * (1 << c2s)))
        simulation_cmd.append('--l2_assoc=16')
        simulation_cmd.append('--cacheline_size=64')
        # simulation_cmd.append('| tee ' + script_file)

        simulation_cmd = ' '.join(simulation_cmd)


        #run the new config
        p = subprocess.Popen(simulation_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).wait()

        #measure the performance
        result = parse_output(stats_file,pattern)
        result_list = result[0].split(' ')
        result_list = [t for t in result_list if t != '']
        sim_sec = float(result_list[1])
        if sim_sec < best_perf:
            best_perf = sim_sec
            best_ci = ci
        
        all_perfs.append(sim_sec)

    print('[ INFO ] trial %d | best: %.6f from option %d' % (ei+1, best_perf*1000000, best_ci))
    print('[ INFO ] all_perf: ', all_perfs)
