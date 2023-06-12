import argparse
import sys
import os

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.params import NULL
from m5.util import addToPath, fatal, warn

addToPath('../')

from ruby import Ruby

from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import ObjectList
from common import MemConfig
from common.FileSystemConfig import config_filesystem
from common.Caches import *
from common.cpu2000 import *

def get_processes(args):
    """Interprets provided args and returns a list of processes"""

    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = args.cmd.split(';')
    if args.input != "":
        inputs = args.input.split(';')
    if args.output != "":
        outputs = args.output.split(';')
    if args.errout != "":
        errouts = args.errout.split(';')
    if args.options != "":
        pargs = args.options.split(';')

    idx = 0
    for wrkld in workloads:
        process = Process(pid = 100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()

        if args.env:
            with open(args.env, 'r') as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    return multiprocesses, 1


parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

if '--ruby' in sys.argv:
    Ruby.define_options(parser)
else:
    print('[ERRO] No Ruby!')
    sys.exit(1)

args = parser.parse_args()

multiprocesses = []
numThreads = 1

if args.cmd:
    multiprocesses, numThreads = get_processes(args)
else:
    print("No workload specified. Exiting!\n", file=sys.stderr)
    sys.exit(1)


(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(args)
CPUClass.numThreads = numThreads

np = args.num_cpus
mp0_path = multiprocesses[0].executable
system = System(cpu = [CPUClass(cpu_id=i) for i in range(np)],
                mem_mode = test_mem_mode,
                mem_ranges = [AddrRange(args.mem_size)],
                cache_line_size = args.cacheline_size,
                workload = SEWorkload.init_compatible(mp0_path))

if numThreads > 1:
    system.multi_thread = True

system.voltage_domain = VoltageDomain(voltage = args.sys_voltage)
system.clk_domain = SrcClockDomain(clock =  args.sys_clock, voltage_domain = system.voltage_domain)
system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain(clock = args.cpu_clock, voltage_domain = system.cpu_voltage_domain)

if args.elastic_trace_en:
    CpuConfig.config_etrace(CPUClass, system.cpu, args)

for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

for i in range(np):
    if len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    system.cpu[i].createThreads()

if args.ruby:
    Ruby.create_system(args, False, system)
    assert(args.num_cpus == len(system.ruby._cpu_ports))
    system.ruby.clk_domain = SrcClockDomain(clock = args.ruby_clock, voltage_domain = system.voltage_domain)
    for i in range(np):
        ruby_port = system.ruby._cpu_ports[i]
        system.cpu[i].createInterruptController()
        ruby_port.connectCpuPorts(system.cpu[i])
else:
    print('[ERRO] Please use Ruby!')
    sys.exit(1)

print('[INFO] # of CPUs:', np)
print('[INFO] CPUClass:', CPUClass)
print('[INFO] test_mem_mode:', test_mem_mode)
print('[INFO] FutureClass:', FutureClass)
print('[INFO] ISA:', args.arm_iset)
print('[INFO] Building Environment:', buildEnv['TARGET_ISA'])
print('[INFO] Memory size:', args.mem_size)
print('[INFO] # of Memory Channels:', args.mem_channels)
print('[INFO] # of Memory Ranks:', args.mem_ranks)
print('[INFO] System clock:', args.sys_clock)
print('[INFO] CPU clock:', args.cpu_clock)
print('[INFO] Ruby clock:', args.ruby_clock)
print('[INFO] System Voltage:', args.sys_voltage)
print('[INFO] CPU Voltage:', system.cpu_voltage_domain.voltage)
if buildEnv['PROTOCOL'] == 'MESI_Three_Level':
    print('[INFO] L0 d cache size:', args.l0d_size)
    print('[INFO] L0 i cache size:', args.l0i_size)
    print('[INFO] L0 d associativity:', args.l0d_assoc)
    print('[INFO] L0 i associativity:', args.l0i_assoc)
    print('[INFO] L1 cache size:', args.l1d_size)
    print('[INFO] L1 associativity:', args.l1d_assoc)
    print('[INFO] L2 cache size:', args.l2_size)
    print('[INFO] L2 associativity:', args.l2_assoc)
elif buildEnv['PROTOCOL'] == 'MOESI_CMP_directory':
    print('[INFO] L1 i cache size:', args.l1i_size)
    print('[INFO] L1 i associativity:', args.l1i_assoc)
    print('[INFO] L1 d cache size:', args.l1d_size)
    print('[INFO] L1 d associativity:', args.l1d_assoc)
    print('[INFO] L2 cache size:', args.l2_size)
    print('[INFO] L2 associativity:', args.l2_assoc)

root = Root(full_system = False, system = system)
Simulation.run(args, root, system, FutureClass)
