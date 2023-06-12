# Copyright (c) 2021 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This script utilizes the X86DemoBoard to run a simple Ubunutu boot. The script
will boot the the OS to login before exiting the simulation.

A detailed terminal output can be found in `m5out/system.pc.com_1.device`.

**Warning:** The X86DemoBoard uses the Timing CPU. The boot may take
considerable time to complete execution.
`configs/example/gem5_library/x86-ubuntu-run-with-kvm.py` can be referenced as
an example of booting Ubuntu with a KVM CPU.

Usage
-----

```
scons build/X86/gem5.opt
./build/X86/gem5.opt configs/example/gem5_library/x86-ubuntu-run.py
```
"""
from gem5.utils.requires import requires
from gem5.components.boards.x86_board import X86Board

#memory
from gem5.components.memory.memory import ChanneledMemory
from gem5.components.memory import dram_interfaces

#cache
from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import MESITwoLevelCacheHierarchy
from gem5.components.cachehierarchies.ruby.moesi_cmp_directory_cache_hierarchy import MOESICMPdirectoryCacheHierarchy

#cpu
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_switchable_processor import SimpleSwitchableProcessor
from gem5.components.processors.simple_processor import SimpleProcessor

#disk
from gem5.resources.resource import Resource, CustomResource

from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent

import os
import re
import optparse
import numpy as np

CPU_TYPE_DICT = {
    "Timing": CPUTypes.TIMING,
    "O3": CPUTypes.O3,
}

MEMORY_TYPE_DICT = {
    "DDR4_2400_8x8": dram_interfaces.ddr4.DDR4_2400_8x8,
    "DDR4_2400_16x4": dram_interfaces.ddr4.DDR4_2400_16x4,
}

parser = optparse.OptionParser()

parser.add_option("--disk-image", type="string")
parser.add_option("--root-partition", type="string", default="/dev/hda1")

parser.add_option("--cpu-type", type="string", default="Timing")
parser.add_option("--num-cpus", type="int", default=4)
parser.add_option("--cpu-clock", type="string", default="2GHz")

parser.add_option("--network", type="string", default="Mesh_nghiant_custom_v2")
parser.add_option("--placement-file", type="string")

parser.add_option("--cache-protocol", type="string", default="MESI_Two_Level")
parser.add_option("--l1d-size", type="string", default="64kB")
parser.add_option("--l1d-assoc", type="int", default=8)
parser.add_option("--l1i-size", type="string", default="64kB")
parser.add_option("--l1i-assoc", type="int", default=8)
parser.add_option("--l2-size", type="string", default="128MB")
parser.add_option("--l2-assoc", type="int", default=16)
parser.add_option("--cacheline-size", type="int", default=64)

parser.add_option("--mem-type", type="string", default="DDR4_2400_16x4")
parser.add_option("--mem-size", type="string", default="4GB")
parser.add_option("--mem-channels", type="int", default=2)

parser.add_option("--command", type="string", default="m5 exit;")

(options, args) = parser.parse_args()

#system components setup
memory = ChanneledMemory(
    MEMORY_TYPE_DICT[options.mem_type],
    options.mem_channels,
    64, #interleaving_size
    size=options.mem_size,
    )

processor = SimpleProcessor(
    cpu_type=CPU_TYPE_DICT[options.cpu_type],
    isa=ISA.X86,
    num_cores=options.num_cpus,
    )

if options.cache_protocol == 'MESI_Two_Level':
    cache_hierarchy = MESITwoLevelCacheHierarchy(
        l1i_size=options.l1i_size,
        l1i_assoc=options.l1i_assoc,
        l1d_size=options.l1d_size,
        l1d_assoc=options.l1d_assoc,
        l2_size=options.l2_size,
        l2_assoc=options.l2_assoc,
        num_l2_banks=options.num_cpus,
    )
elif options.cache_protocol == 'MOESI_CMP_directory':
    cache_hierarchy = MOESICMPdirectoryCacheHierarchy(
        l1i_size=options.l1i_size,
        l1i_assoc=options.l1i_assoc,
        l1d_size=options.l1d_size,
        l1d_assoc=options.l1d_assoc,
        l2_size=options.l2_size,
        l2_assoc=options.l2_assoc,
        network_type=options.network,
        placement_file=options.placement_file
    )


# Here we setup the board
board = X86Board(
    clk_freq=options.cpu_clock,
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
    )

board.set_kernel_disk_workload(
    kernel=Resource(
        "x86-linux-kernel-4.19.83",
        ),
    disk_image=CustomResource(
        options.disk_image,
        metadata={"root_partition": options.root_partition}),
    readfile_contents=options.command,
)

simulator = Simulator(board=board)
simulator.run()
