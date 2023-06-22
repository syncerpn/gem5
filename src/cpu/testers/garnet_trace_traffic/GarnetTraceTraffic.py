# Copyright (c) 2016 Georgia Institute of Technology
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

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *

class GarnetTraceTraffic(ClockedObject):
    type = 'GarnetTraceTraffic'
    cxx_header = \
        "cpu/testers/garnet_trace_traffic/GarnetTraceTraffic.hh"
    cxx_class = 'gem5::GarnetTraceTraffic'

    block_offset = Param.Int(6, "block offset in bits")
    memory_size = Param.Int(65536, "memory size")
    sim_cycles = Param.UInt64(1000, "Number of simulation cycles")
    num_packets_max = Param.Int(-1, "Max number of packets to send. \
                        Default is to keep sending till simulation ends")
    response_limit = Param.Cycles(100000000, "Cycles before exiting \
                                            due to lack of progress")
    test = RequestPort("Port to the memory system to test")
    system = Param.System(Parent.any, "System we belong to")
    trace = Param.String("", "trace file")
    cpu_id = Param.Int(0, "CPU ID")