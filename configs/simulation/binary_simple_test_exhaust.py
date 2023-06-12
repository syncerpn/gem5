import os
import re
import optparse
import numpy as np
import signal
import subprocess
import itertools

# np.random.seed(1)

parser = optparse.OptionParser()

parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/X86_MOESI_CMP_directory_opt/gem5.opt", help="gem5 build")
parser.add_option("-s",  "--sim-sys",        type="string", default="/home/nghiant/git/gem5/configs/simulation/sim_system.py", help="simulation system")
parser.add_option("-n",  "--num-cpus",       type="int",    default=4, help="number of cpus")
parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_test_trace_test/", help="output dir")

parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.gcc-openmp/bin/blackscholes", help="workload")
parser.add_option("-a",  "--workload-args",  type="string", default="2 input_blackscholes_test/in_4.txt input_blackscholes_test/out_4.txt", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 input_blackscholes_simdev/in_16.txt input_blackscholes_simdev/out_16.txt", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="32 input_blackscholes_simsmall/in_4K.txt input_blackscholes_simsmall/out_4K.txt", help="workload arguments")
#==================

parser.add_option("-p",  "--protocol",       type="string", default="moesi", help="protocol")
parser.add_option("-v",  "--mesh-version",   type="int",    default=2, help="mesh implementation version [1, 2]")
parser.add_option("-r",  "--mesh-row",       type="int",    default=2, help="mesh max row")
parser.add_option("-c",  "--mesh-col",       type="int",    default=4, help="mesh max column")

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
    all_ports = list(range(n_port))
    np.random.shuffle(all_ports)
    occupied = all_ports[:n_device]
    return occupied

#hyper-param settings
n_port_per_node = 1
pattern = ['simSeconds']

#prepare command
presets_occupied = [
                    # [3, 5, 7, 2, 4, 6],
                    # [6, 1, 3, 4, 0, 7],
                    # [1, 3, 0, 6, 7, 4],
                    # [7, 6, 4, 0, 3, 5],
                    # [3, 5, 7, 4, 6, 0],
                    # [4, 7, 3, 1, 5, 2],
                    # [3, 1, 7, 5, 2, 6],
                    # [5, 6, 2, 4, 1, 0],
                    # [5, 1, 4, 7, 6, 2],
                    # [2, 7, 1, 3, 6, 5], #optimal estimated by code
                    # [7, 3, 5, 4, 0, 6], #worst estimated by code
                    # [33,10,63,57,18,24,44,55,13,23,42,46,34,16,53,49,52,12,17,25,58,50,21,54,32,20,11,38,9,5,37,36,22,31],
                    # [36,51,29,40,61,19,34,50,21,23,3,20,49,24,17,44,31,59,42,62,10,12,22,52,13,37,11,45,25,5,15,26,28,27], #code estimated
                    # [44, 25, 35, 10, 58, 54, 45, 42, 28, 27, 7, 19, 26, 4, 29, 20, 40, 32, 37, 52, 41, 50, 51, 63, 47, 30, 2, 38, 15, 13, 12, 59, 36, 43], #code estimated
                    # [56, 62, 39, 10, 19, 31, 58, 36, 49, 24, 1, 42, 2, 59, 52, 57, 13, 6, 38, 15, 3, 54, 12, 17, 16, 55, 27, 9, 46, 22, 14, 30, 7, 48], #code estimated worst
                    # [28, 35, 43, 0, 42, 59, 54, 15, 10, 27, 44, 39, 24, 20, 11, 7, 53, 45, 4, 34, 17, 3, 23, 19, 38, 56, 46, 49, 21, 14, 26, 1, 36, 37],
                    ]


for preset_occupied in presets_occupied:
    trial_out_dir = options.out_dir + 'test_exhaust/'

    os.makedirs(trial_out_dir, exist_ok=True)

    script_file = trial_out_dir + 'script'
    stats_file  = trial_out_dir + 'stats'
    placement_file = trial_out_dir + 'placement'
    rubygentrace_file =  trial_out_dir + 'rubygentracex_4cpu_meshv2_' + '-'.join(list(map(str, preset_occupied))) + '.out'

    simulation_cmd  = [options.gem5_bin]
    
    #nghiant: trace
    simulation_cmd.append('--debug-flags=nghiant_RubyNetwork')
    simulation_cmd.append('--debug-file=' + rubygentrace_file)
    #nghiant: trace

    simulation_cmd.append('--outdir=' + trial_out_dir)
    simulation_cmd.append('--stats-file=' + stats_file)
    simulation_cmd.append(options.sim_sys)
    simulation_cmd.append('--cmd=' + options.workload)
    simulation_cmd.append('--options=' + '"' + options.workload_args + '"')
    simulation_cmd.append('--ruby')
    simulation_cmd.append('--network=garnet')
    
    if options.mesh_version == 1:
        simulation_cmd.append('--topology=Mesh_nghiant_custom')
    elif options.mesh_version == 2:
        simulation_cmd.append('--topology=Mesh_nghiant_custom_v2')

    simulation_cmd.append('--placement-file=' + placement_file)
    simulation_cmd.append('--num-cpus=' + str(options.num_cpus))

    simulation_cmd.append('--cpu-type=DerivO3CPU')

    simulation_cmd.append('--cpu-clock=2GHz')
    simulation_cmd.append('--mem-type=DDR4_2400_16x4')
    simulation_cmd.append('--mem-size=4GB')
    simulation_cmd.append('--mem-channels=4')
    simulation_cmd.append('--mem-ranks=2')
    simulation_cmd.append('--l1i_size=64kB')
    simulation_cmd.append('--l1i_assoc=8')
    simulation_cmd.append('--l1d_size=64kB')
    simulation_cmd.append('--l1d_assoc=8')
    simulation_cmd.append('--l2_size=128MB')
    simulation_cmd.append('--l2_assoc=16')
    simulation_cmd.append('--cacheline_size=64')
    simulation_cmd.append('--cacheline_size=64')
    simulation_cmd.append('| tee ' + script_file)

    simulation_cmd = ' '.join(simulation_cmd)
    print(simulation_cmd)

    #update input mesh config
    if options.protocol == 'moesi':
        n_port = options.mesh_row * options.mesh_col * n_port_per_node
        device_name = ['L1Cache_Controller'] * options.num_cpus + ['L2Cache_Controller'] + ['Directory_Controller']
        device_id = list(range(options.num_cpus)) + [0] + [0]
        n_device = len(device_name)
        assert(n_device <= n_port)

        occupied = random_device_placement(n_device, n_port)

        occupied = preset_occupied

        print('[ INFO ] occupied', occupied)

    else:
        print('unknown protocol')
        assert(0)

    with open(placement_file, 'w') as f:
        lines = [str(options.mesh_row) + ' ' + str(options.mesh_col)]
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
    p = subprocess.Popen(simulation_cmd, shell=True).wait()

    #measure the performance
    result = parse_output(stats_file,pattern)
    result_list = result[0].split(' ')
    result_list = [t for t in result_list if t != '']
    sim_sec = float(result_list[1])
        
    print('[ INFO ] preset %s | binary execution time: %.6f' % (''.join(list(map(str,preset_occupied))), sim_sec))

    #update the estimation model/search algorithm
