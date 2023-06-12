import os
import re
import optparse
import numpy as np
import signal
import subprocess
import itertools

# np.random.seed(1)

parser = optparse.OptionParser()

parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/Garnet_trace/gem5.opt", help="gem5 build")

parser.add_option("-s",  "--sim-sys",        type="string", default="/home/nghiant/git/gem5/configs/simulation/sim_network.py", help="simulation system")
parser.add_option("--simulated-protocol", type="string", default="moesi")
parser.add_option("-n",  "--num-cpus",       type="int",    default=32, help="number of cpus")
parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/trace_gn_test/", help="output dir")

parser.add_option("-p",  "--protocol",       type="string", default="moesi", help="protocol")
parser.add_option("-r",  "--mesh-row",       type="int",    default=8, help="mesh max row")
parser.add_option("-c",  "--mesh-col",       type="int",    default=8, help="mesh max column")

parser.add_option("--trace", type="string", default="sample_trace_gn_packet_2")
# parser.add_option("--trace", type="string", default="sample_trace_gn_packet_psize")

parser.add_option("--sim-cycles", type="int", default=0,
                    help="Number of simulation cycles")

parser.add_option("--num-packets-max", type="int", default=-1,
                    help="Stop injecting after --num-packets-max.\
                        Set to -1 to disable.")

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
pattern = ['system.ruby.network.average_flit_latency']
# pattern = ['system.ruby.network.average_packet_latency']

#prepare command
best_perf = 99999
for ei in range(1):

    presets_occupied = [
    # [33,10,63,57,18,24,44,55,13,23,42,46,34,16,53,49,52,12,17,25,58,50,21,54,32,20,11,38,9,5,37,36,22,31],
    [61, 62, 16, 7, 27, 8, 3, 1, 49, 41, 36, 56, 44, 60, 13, 5, 12, 54, 37, 28, 17, 9, 24, 58, 4, 2, 31, 20, 50, 52, 55, 42, 63, 33],
    [38, 51, 55, 35, 47, 61, 53, 22, 14, 21, 58, 40, 6, 42, 31, 48, 36, 10, 23, 34, 46, 13, 49, 60, 18, 57, 20, 24, 43, 63, 50, 25, 12, 54],
    [0, 59, 58, 5, 46, 54, 3, 20, 29, 15, 26, 41, 43, 62, 60, 37, 12, 55, 56, 13, 48, 45, 21, 22, 50, 10, 4, 33, 52, 6, 49, 30, 11, 31],
    [10, 56, 45, 25, 40, 3, 43, 32, 31, 7, 11, 35, 23, 55, 54, 14, 26, 0, 36, 50, 4, 48, 60, 63, 62, 21, 9, 47, 6, 53, 42, 33, 52, 1],
    [2, 60, 53, 59, 4, 54, 44, 63, 41, 10, 29, 12, 51, 61, 37, 17, 57, 49, 56, 9, 6, 13, 11, 55, 16, 28, 39, 5, 46, 32, 50, 0, 34, 40],
    [54, 27, 32, 11, 38, 14, 16, 44, 31, 52, 45, 39, 30, 51, 61, 49, 58, 50, 40, 48, 42, 47, 56, 9, 7, 41, 36, 2, 55, 60, 25, 10, 1, 4],
    [34, 36, 56, 6, 44, 41, 22, 38, 25, 32, 14, 29, 7, 28, 59, 19, 37, 0, 61, 58, 47, 30, 27, 60, 33, 21, 49, 31, 50, 46, 1, 51, 62, 13],
    [28, 27, 55, 24, 45, 59, 50, 40, 49, 9, 41, 37, 48, 8, 15, 60, 14, 54, 62, 11, 7, 35, 18, 20, 25, 51, 61, 56, 17, 2, 63, 46, 22, 58],
    [13, 40, 57, 10, 50, 45, 18, 8, 16, 29, 12, 38, 2, 42, 7, 33, 34, 5, 46, 24, 61, 36, 44, 49, 58, 59, 30, 43, 48, 9, 19, 54, 62, 47],
    [35, 31, 4, 43, 0, 15, 44, 21, 5, 26, 41, 3, 28, 60, 49, 17, 47, 54, 57, 24, 59, 58, 12, 63, 61, 9, 18, 48, 29, 42, 46, 6, 52, 19],
    ]

    for preset_occupied in presets_occupied:
        num_fake_cpus = 0
        if options.simulated_protocol == "moesi":
            num_fake_cpus = 2 #L2 cache
            num_dirs = options.mesh_row * options.mesh_col

        trial_out_dir = options.out_dir + 'test' + str(ei) + '/'

        os.makedirs(trial_out_dir, exist_ok=True)

        script_file = trial_out_dir + 'script'
        stats_file  = trial_out_dir + 'stats'
        placement_file = trial_out_dir + 'placement'
        rubygentrace_file =  trial_out_dir + 'retrace_' + ''.join(list(map(str, preset_occupied)))

        converted_trace = options.trace + '_converted.txt'

        simulation_cmd  = [options.gem5_bin]
        simulation_cmd.append('--debug-flags=nghiant_RubyNetwork')
        simulation_cmd.append('--debug-file=' + rubygentrace_file)

        simulation_cmd.append('--outdir=' + trial_out_dir)
        simulation_cmd.append('--stats-file=' + stats_file)
        simulation_cmd.append(options.sim_sys)
        simulation_cmd.append('--ruby')
        simulation_cmd.append('--network=garnet')
        simulation_cmd.append('--topology=Mesh_gn')
        simulation_cmd.append('--placement-file=' + placement_file)
        simulation_cmd.append('--trace=' + converted_trace)
        simulation_cmd.append('--num-cpus=' + str(options.num_cpus + num_fake_cpus))
        simulation_cmd.append('--num-dirs=' + str(num_dirs))
        simulation_cmd.append('--cpu-clock=2GHz')
        simulation_cmd.append('--mem-type=DDR4_2400_16x4')
        simulation_cmd.append('--mem-size=340MB')
        simulation_cmd.append('--mem-channels=1')
        simulation_cmd.append('--mem-ranks=2')
        simulation_cmd.append('--l1i_size=64kB')
        simulation_cmd.append('--l1i_assoc=8')
        simulation_cmd.append('--l1d_size=64kB')
        simulation_cmd.append('--l1d_assoc=8')
        simulation_cmd.append('--l2_size=128MB')
        simulation_cmd.append('--l2_assoc=16')
        simulation_cmd.append('--cacheline_size=64')
        simulation_cmd.append('--sim-cycles=800000000')
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
            lines = [f'{options.mesh_row} {options.mesh_col}']
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

        #nghiant: need to convert original trace, now with placement info into
        #<cycle> <cpu_id> <router_dest_id> <vnet> format for correct behavior
        with open(options.trace, 'r') as f:
            with open(converted_trace, 'w') as nf:
                data = f.readline().strip('\n')
                while data:
                    cycle, s, d, v = map(int, data.split(' '))
                    new_d = occupied[d]
                    nf.write(' '.join([str(cycle),str(s),str(new_d),str(v)])+'\n')
                        
                    data = f.readline().strip('\n')

        #nghiant: need to convert original trace, now with placement info into
        #<cycle> <cpu_id> <router_dest_id> <vnet> format for correct behavior
        # with open(options.trace, 'r') as f:
        #     with open(converted_trace, 'w') as nf:
        #         data = f.readline().strip('\n')
        #         while data:
        #             cycle, s, d, v, p = map(int, data.split(' '))
        #             new_d = occupied[d]
        #             nf.write(' '.join([str(cycle),str(s),str(new_d),str(v),str(p)])+'\n')
                        
        #             data = f.readline().strip('\n')


        #run the new config
        p = subprocess.Popen(simulation_cmd, shell=True).wait()

        #measure the performance
        result = parse_output(stats_file,pattern)
        result_list = result[0].split(' ')
        result_list = [t for t in result_list if t != '']
        sim_sec = float(result_list[1])
        if sim_sec < best_perf:
            best_perf = sim_sec
            
        print('[ INFO ] trial %d | flit network latency: %.6f' % (ei+1, sim_sec))


        #update the estimation model/search algorithm
