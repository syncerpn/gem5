import os
import re
import optparse
import numpy as np
import signal
import subprocess
import itertools

# np.random.seed(1)

parser = optparse.OptionParser()

# parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/X86_MOESI_CMP_directory_fast/gem5.fast", help="gem5 build")
parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/X86_MOESI_CMP_directory_opt/gem5.opt", help="gem5 build")
# parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/ARM_MOESI_CMP_directory_opt/gem5.opt", help="gem5 build")
# parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/ARM_MOESI_CMP_directory_fast/gem5.fast", help="gem5 build")
# parser.add_option(       "--gem5-bin",       type="string", default="/home/nghiant/git/gem5/build/RISCV_MOESI_CMP_directory_opt/gem5.opt", help="gem5 build")

parser.add_option("-s",  "--sim-sys",        type="string", default="/home/nghiant/git/gem5/configs/simulation/sim_system.py", help="simulation system")
parser.add_option("-n",  "--num-cpus",       type="int",    default=32, help="number of cpus")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_test_trace_test_riscv/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_test_comtrace_fast_arm/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_test_comtrace_slow_arm/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_735406/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_613407/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_357246/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_130674/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_357460/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_764035/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_473152/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_317526/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_514762/", help="output dir")
# parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_comtrace_more_271365/", help="output dir")
parser.add_option("-o",  "--out-dir",        type="string", default="/home/nghiant/git/gem5/binary_trace_gn/", help="output dir")

parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.gcc-openmp/bin/blackscholes", help="workload")
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.aarch64/bin/blackscholes", help="workload")
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.riscv64/bin/blackscholes", help="workload")

# parser.add_option("-a",  "--workload-args",  type="string", default="2 input_blackscholes_test/in_4.txt input_blackscholes_test/out_4.txt", help="workload arguments")
parser.add_option("-a",  "--workload-args",  type="string", default="4 input_blackscholes_simdev/in_16.txt input_blackscholes_simdev/out_16.txt", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 input_blackscholes_simsmall/in_4K.txt input_blackscholes_simsmall/out_4K.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/bodytrack/inst/amd64-linux.gcc-openmp/bin/bodytrack", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="input_bodytrack_test/sequenceB_1 4 1 5 1 0 4", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="input_bodytrack_simdev/sequenceB_1 4 1 100 3 0 32", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="input_bodytrack_simsmall/sequenceB_1 4 1 1000 5 0 4", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/darknet/darknet_x86", help="workload")
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/darknet/darknet_riscv64", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="kernel", help="workload arguments")

#PARSEC freqmine
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/freqmine/inst/amd64-linux.gcc-openmp/bin/freqmine", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="input_freqmine_test/T10I4D100K_3.dat 1", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="input_freqmine_simdev/T10I4D100K_1k.dat 3", help="workload arguments")
# parser.add_option("-a",  "--workload-args",  type="string", default="input_freqmine_simsmall/kosarak_250k.dat 220", help="workload arguments")


# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/two/blackscholes/obj/aarch64-linux.gcc-hooks/blackscholes", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 workload/in_4.txt workload/out_4.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/arm/ferret_pthreads/inst/aarch64-linux.gcc-pthreads/bin/ferret", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="workload/two/ferret/corel lsh workload/two/ferret/queries 5 5 4 output.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/arm/ferret/inst/aarch64-linux.gcc/bin/ferret", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="workload/two/ferret/corel lsh workload/two/ferret/queries 5 5 4 output.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/arm/blackscholes_openmp", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="32 workload/in_512.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec_blackscholes/blackscholes_openmp", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 workload/parsec_blackscholes/in_4.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec_blackscholes/blackscholes_openmp_x86", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 workload/parsec_blackscholes/in_4.txt", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec_canneal/canneal_pthread", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 5 100 workload/parsec_canneal/10.nets 1", help="workload arguments")

#==================
#PARSEC canneal: HANGED (TOOK TOO LONG TIME)
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/kernels/canneal/inst/aarch64-linux.gcc-pthreads/bin/canneal", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="3 5 2 10.nets 1", help="workload arguments")

#PARSEC swaptions: OK
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/apps/swaptions/inst/aarch64-linux.gcc-pthreads/bin/swaptions", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-ns 8 -sm 5 -nt 4", help="workload arguments")

#PARSEC dedup: FAILED
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/kernels/dedup/inst/aarch64-linux.gcc-pthreads/bin/dedup", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-c -p -v -t 2 -i test.dat -o output.dat.ddp", help="workload arguments")

#PARSEC streamcluster:
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/kernels/streamcluster/inst/aarch64-linux.gcc-pthreads/bin/streamcluster", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="2 5 1 10 10 5 none output.txt 3", help="workload arguments")

#PARSEC blackscholes: OK
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/apps/blackscholes/inst/aarch64-linux.gcc-openmp/bin/blackscholes", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="4 workload/parsec_blackscholes/in_4.txt output.txt", help="workload arguments")

#PARSEC bodytrack

#PARSEC facesim
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/apps/facesim/inst/aarch64-linux.gcc-pthreads/bin/facesim", help="workload")
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/facesim/inst/amd64-linux.gcc-pthreads/bin/facesim", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-timing -threads 3", help="workload arguments")

#PARSEC ferret
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/parsec-3.0/pkgs/apps/ferret/inst/amd64-linux.gcc/bin/ferret", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="corel lsh queries 5 5 1 output.txt", help="workload arguments")

#PARSEC fluidanimate
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/apps/fluidanimate/inst/aarch64-linux.gcc-pthreads/bin/fluidanimate", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="2 1 /hdd/parsec-3.0/pkgs/apps/fluidanimate/inputs/in_5K.fluid out.fluid", help="workload arguments")

#PARSEC freqmine
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/apps/freqmine/inst/aarch64-linux.gcc-openmp/bin/freqmine", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="T10I4D100K_3.dat 1", help="workload arguments")

#PARSEC raytrace

#PARSEC swaptions

#PARSEC vips

#PARSEC x264
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/parsec-3.0/pkgs/apps/x264/inst/aarch64-linux.gcc-pthreads/bin/x264", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 --threads 4 -o eledream.264 eledream_32x18_1.y4m", help="workload arguments")


# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/darknet/darknet_arm", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="kernel", help="workload arguments")

#darknet_x86 small cifar test: OK
# parser.add_option("-w",  "--workload",       type="string", default="/home/nghiant/git/gem5/workload/darknet/darknet_x86", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="classifier test /home/nghiant/git/gem5/workload/darknet/data/cifar.data /home/nghiant/git/gem5/workload/darknet/cfg/cifar.cfg /home/nghiant/git/gem5/workload/darknet/backup/cifar.weights /home/nghiant/git/gem5/workload/darknet/data/1045_dog.png", help="workload arguments")


# parser.add_option("-w",  "--workload",       type="string", default="/usr/bin/mpirun", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-np 4 /home/nghiant/git/gem5/workload/hpcc/hpcc", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/usr/bin/mpirun", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-np 4 /home/nghiant/git/gem5/workload/hpcc/hpcc", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/usr/bin/mpirun", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-np 4 /home/nghiant/git/gem5/workload/hpl/bin/linux/xhpl", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/usr/bin/mpirun", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="-np 4 /home/nghiant/git/gem5/workload/custom/mpi_hello_world", help="workload arguments")

# parser.add_option("-w",  "--workload",       type="string", default="/hdd/wangren/CPU2006_RISCV/benchspec/CPU2006/401.bzip2/run/run_base_ref_riscv.0000/bzip2_ref", help="workload")
# parser.add_option("-w",  "--workload",       type="string", default="/hdd/wangren/CPU2006/benchspec/CPU2006/401.bzip2/run/run_base_ref_gcc41-64bit.0000/bzip2_ref", help="workload")
# parser.add_option("-a",  "--workload-args",  type="string", default="", help="workload arguments")

#==================

parser.add_option("-p",  "--protocol",       type="string", default="moesi", help="protocol")
parser.add_option("-r",  "--mesh-row",       type="int",    default=8, help="mesh max row")
parser.add_option("-c",  "--mesh-col",       type="int",    default=8, help="mesh max column")

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
best_perf = 99999
for ei in range(1):

    presets_occupied = [
                        # [0,1,2,3],
                        # [2,1,0,3],
                        [33,10,63,57,18,24,44,55,13,23,42,46,34,16,53,49,52,12,17,25,58,50,21,54,32,20,11,38,9,5,37,36,22,31],
                        # [61, 62, 16, 7, 27, 8, 3, 1, 49, 41, 36, 56, 44, 60, 13, 5, 12, 54, 37, 28, 17, 9, 24, 58, 4, 2, 31, 20, 50, 52, 55, 42, 63, 33],
                        # [38, 51, 55, 35, 47, 61, 53, 22, 14, 21, 58, 40, 6, 42, 31, 48, 36, 10, 23, 34, 46, 13, 49, 60, 18, 57, 20, 24, 43, 63, 50, 25, 12, 54],
                        # [0, 59, 58, 5, 46, 54, 3, 20, 29, 15, 26, 41, 43, 62, 60, 37, 12, 55, 56, 13, 48, 45, 21, 22, 50, 10, 4, 33, 52, 6, 49, 30, 11, 31],
                        # [10, 56, 45, 25, 40, 3, 43, 32, 31, 7, 11, 35, 23, 55, 54, 14, 26, 0, 36, 50, 4, 48, 60, 63, 62, 21, 9, 47, 6, 53, 42, 33, 52, 1],
                        # [2, 60, 53, 59, 4, 54, 44, 63, 41, 10, 29, 12, 51, 61, 37, 17, 57, 49, 56, 9, 6, 13, 11, 55, 16, 28, 39, 5, 46, 32, 50, 0, 34, 40],
                        # [54, 27, 32, 11, 38, 14, 16, 44, 31, 52, 45, 39, 30, 51, 61, 49, 58, 50, 40, 48, 42, 47, 56, 9, 7, 41, 36, 2, 55, 60, 25, 10, 1, 4],
                        # [34, 36, 56, 6, 44, 41, 22, 38, 25, 32, 14, 29, 7, 28, 59, 19, 37, 0, 61, 58, 47, 30, 27, 60, 33, 21, 49, 31, 50, 46, 1, 51, 62, 13],
                        # [28, 27, 55, 24, 45, 59, 50, 40, 49, 9, 41, 37, 48, 8, 15, 60, 14, 54, 62, 11, 7, 35, 18, 20, 25, 51, 61, 56, 17, 2, 63, 46, 22, 58],
                        # [13, 40, 57, 10, 50, 45, 18, 8, 16, 29, 12, 38, 2, 42, 7, 33, 34, 5, 46, 24, 61, 36, 44, 49, 58, 59, 30, 43, 48, 9, 19, 54, 62, 47],
                        # [35, 31, 4, 43, 0, 15, 44, 21, 5, 26, 41, 3, 28, 60, 49, 17, 47, 54, 57, 24, 59, 58, 12, 63, 61, 9, 18, 48, 29, 42, 46, 6, 52, 19],
                        # [4,6,5,2,3,0],
                        # [6,5,1,7,2,3],

                        # [7,3,5,4,0,6],
                        # [2,7,1,3,6,5],
                        #xx
                        # [7,3,5,4,0,6],
                        # [6,1,3,4,0,7],
                        # [3,5,7,2,4,6],
                        # [1,3,0,6,7,4],
                        # [3,5,7,4,6,0],
                        # [7,6,4,0,3,5],
                        # [4,7,3,1,5,2],
                        # [3,1,7,5,2,6],
                        # [5,1,4,7,6,2],
                        # [2,7,1,3,6,5],
                        # [5,6,2,4,1,0],
                        #xx

                        # [0,1,3],
                        # [0,2,1],
                        # [0,2,3],
                        # [0,3,1],
                        # [0,3,2],
                        # [1,0,2],
                        # [1,0,3],
                        # [1,0,2],
                        # [1,0,3],
                        # [1,3,0],
                        # [1,3,2],
                        # [2,0,1],
                        # [2,0,3],
                        # [2,1,0],
                        # [2,1,3],
                        # [2,3,0],
                        # [2,3,1],
                        # [3,0,1],
                        # [3,0,2],
                        # [3,1,0],
                        # [3,1,2],
                        # [3,2,0],
                        # [3,2,1],
                        ]


    for preset_occupied in presets_occupied:
        trial_out_dir = options.out_dir + 'test' + str(ei) + '/'

        os.makedirs(trial_out_dir, exist_ok=True)

        script_file = trial_out_dir + 'script'
        stats_file  = trial_out_dir + 'stats'
        placement_file = trial_out_dir + 'placement'
        trace_file = trial_out_dir + 'trace_exec_short_1_' + ''.join(list(map(str, preset_occupied))) + '.out'
        protocoltrace_file = trial_out_dir + 'protocoltrace_2cpu_' + ''.join(list(map(str, preset_occupied))) + '.out'
        rubygentrace_file =  trial_out_dir + 'rubygentracex_2cpu_' + ''.join(list(map(str, preset_occupied))) + '.out'

        simulation_cmd  = [options.gem5_bin]
        
        #nghiant: trace
        simulation_cmd.append('--debug-flags=nghiant_RubyNetwork')
        simulation_cmd.append('--debug-file=' + rubygentrace_file)

        # simulation_cmd.append('--debug-flags=ProtocolTrace')
        # simulation_cmd.append('--debug-file=' + protocoltrace_file)

        # simulation_cmd.append('--debug-flags=Exec')
        # simulation_cmd.append('--debug-file=' + trace_file)
        #nghiant: trace

        simulation_cmd.append('--outdir=' + trial_out_dir)
        simulation_cmd.append('--stats-file=' + stats_file)
        simulation_cmd.append(options.sim_sys)
        simulation_cmd.append('--cmd=' + options.workload)
        simulation_cmd.append('--options=' + '"' + options.workload_args + '"')
        simulation_cmd.append('--ruby')
        simulation_cmd.append('--network=garnet')
        # simulation_cmd.append('--network=simple')
        simulation_cmd.append('--topology=Mesh_nghiant_custom')
        simulation_cmd.append('--placement-file=' + placement_file)
        simulation_cmd.append('--num-cpus=' + str(options.num_cpus))

        simulation_cmd.append('--cpu-type=DerivO3CPU')
        # simulation_cmd.append('--cpu-type=O3_ARM_v7a_3')
        # simulation_cmd.append('--arm-iset=aarch64')

        simulation_cmd.append('--cpu-clock=2GHz')
        simulation_cmd.append('--mem-type=DDR4_2400_16x4')
        simulation_cmd.append('--mem-size=4GB')
        simulation_cmd.append('--mem-channels=1')
        simulation_cmd.append('--mem-ranks=2')
        simulation_cmd.append('--l1i_size=64kB')
        simulation_cmd.append('--l1i_assoc=8')
        simulation_cmd.append('--l1d_size=64kB')
        simulation_cmd.append('--l1d_assoc=8')
        simulation_cmd.append('--l2_size=128MB')
        simulation_cmd.append('--l2_assoc=16')
        simulation_cmd.append('--cacheline_size=64')
        simulation_cmd.append('--cacheline_size=64')
        # simulation_cmd.append('--maxinsts=500000000')
        # simulation_cmd.append('--warmup-insts=10000')
        simulation_cmd.append("--num-dirs=1")
        simulation_cmd.append("--num-l2caches=1")
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
            # occupied = [0,4,3,2,1,5]         #trace_exe.out
            # occupied = [4,0,2,1,7,5]         #trace_exe2.out
            occupied = preset_occupied

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

        #run the new config
        p = subprocess.Popen(simulation_cmd, shell=True).wait()

        #measure the performance
        result = parse_output(stats_file,pattern)
        result_list = result[0].split(' ')
        result_list = [t for t in result_list if t != '']
        sim_sec = float(result_list[1])
        if sim_sec < best_perf:
            best_perf = sim_sec
            
        print('[ INFO ] trial %d | binary execution time: %.6f | best_perf %f' % (ei+1, sim_sec, best_perf*1000000))


        #update the estimation model/search algorithm
