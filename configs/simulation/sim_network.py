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

parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

if '--ruby' in sys.argv:
    Ruby.define_options(parser)
else:
    print('[ERRO] No Ruby!')
    sys.exit(1)

args = parser.parse_args()

np = args.num_cpus
# print('heelo', args.num_fake_cpus)
cpus = [ GarnetTraceTraffic(
                     trace=args.trace,
                     num_packets_max=args.num_packets_max,
                     sim_cycles=args.sim_cycles,
                     cpu_id=i) \
         for i in range(args.num_cpus) ]

system = System(cpu = cpus,
                mem_ranges = [AddrRange(args.mem_size)])

system.voltage_domain = VoltageDomain(voltage = args.sys_voltage)
system.clk_domain = SrcClockDomain(clock =  args.sys_clock, voltage_domain = system.voltage_domain)
system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain(clock = args.cpu_clock, voltage_domain = system.cpu_voltage_domain)

for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

if args.ruby:
    Ruby.create_system(args, False, system)
    assert(args.num_cpus == len(system.ruby._cpu_ports))
    system.ruby.clk_domain = SrcClockDomain(clock = args.ruby_clock, voltage_domain = system.voltage_domain)
    for i in range(np):
        system.cpu[i].test = system.ruby._cpu_ports[i].in_ports
else:
    print('[ERRO] Please use Ruby!')
    sys.exit(1)

print('[INFO] # of CPUs:', np)
print('[INFO] Building Environment:', buildEnv['TARGET_ISA'])
print('[INFO] Memory size:', args.mem_size)
print('[INFO] # of Memory Channels:', args.mem_channels)
print('[INFO] # of Memory Ranks:', args.mem_ranks)
print('[INFO] System clock:', args.sys_clock)
print('[INFO] CPU clock:', args.cpu_clock)
print('[INFO] Ruby clock:', args.ruby_clock)
print('[INFO] System Voltage:', args.sys_voltage)
print('[INFO] CPU Voltage:', system.cpu_voltage_domain.voltage)

root = Root(full_system = False, system = system)
# Simulation.run(args, root, system, FutureClass)

root.system.mem_mode = 'timing'

# Not much point in this being higher than the L1 latency
m5.ticks.setGlobalFrequency('1ps')

# instantiate configuration
m5.instantiate()

# simulate until program terminates
exit_event = m5.simulate(args.abs_max_tick)

print('Exiting @ tick', m5.curTick(), 'because', exit_event.getCause())
