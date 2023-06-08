# Copyright (c) 2010 Advanced Micro Devices, Inc.
#               2016 Georgia Institute of Technology
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
#nghiant:custom mesh network
from __future__ import print_function
from __future__ import absolute_import

from m5.params import *
from m5.objects import *

from common import FileSystemConfig

from topologies.BaseTopology import SimpleTopology

# Creates a generic Mesh assuming an equal number of cache
# and directory controllers.
# XY routing is enforced (using link weights)
# to guarantee deadlock freedom.

from m5.defines import buildEnv

import re
def parse_placement(file):
    f = open(file, "r")
    a = f.read()

    #nghiant_220525: row and col are specified in the placement file as well
    rc, a = a.split('\n', 1)
    row, col = list(map(int, rc.split(' ')))
    #nghiant_end

    z = re.split('[ \t\n\r]', a)
    assert(len(z) % 4 == 0)
    placement = [{"type":z[i], "id":int(z[i+1]), "row":int(z[i+2]), "col":int(z[i+3])} for i in range(0, len(z), 4)]
    print("[INFO] placement:")
    for i in placement:
        print("%s_%d\t[%d, %d]" % (i["type"], i["id"], i["row"], i["col"]))
    return row, col, placement

def validate_placement(placement, max_per_router, spec_row, spec_col):
    if buildEnv['PROTOCOL'] == 'Garnet_trace':
        #nghiant: validate if exceeding max_per_router; L0s are not counted
        filtered_placement = [entity for entity in placement]
        num_rows = max([i["row"] for i in filtered_placement]) + 1
        num_cols = max([i["col"] for i in filtered_placement]) + 1
        for i in range(num_rows):
            for j in range(num_cols):
                counter = 0
                for entity in filtered_placement:
                    if entity["row"] == i and entity["col"] == j:
                        counter += 1
                if counter > max_per_router:
                    print("[ERRO] exceeded maximum nodes per router (connected: %d/max: %d at [%d, %d])" % (counter, max_per_router, i, j))
                    assert(0)
                if counter == 0:
                    print("[WARN] found an isolated node (connected: %d at [%d, %d])" % (counter, i, j))

        assert num_rows <= spec_row, '[ERRO] specified number of rows is not compatible with the mesh config'
        assert num_cols <= spec_col, '[ERRO] specified number of cols is not compatible with the mesh config'
        if num_rows < spec_row or num_cols < spec_col:
            print('[WARN] mesh can be tighten: (%d x %d) -> (%d x %d)' % (spec_row, spec_col, num_rows, num_cols))
    else:
        assert 0, '[ERRO] unsupported buildEnv PROTOCOL \'%s\'; please implement/specify details for it' % buildEnv['PROTOCOL']

class Mesh_gn(SimpleTopology):
    description='Mesh_gn'

    def __init__(self, controllers):
        assert buildEnv['PROTOCOL'] == 'Garnet_trace'
        self.nodes = controllers

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        num_rows, num_cols, placement = parse_placement(options.placement_file)
        nodes = self.nodes

        nn = len(nodes)
        np = len(placement)
        assert(nn >= np)

        #nghiant: for CMN-600, upto 2 nodes can be connected to a router
        cntrls_per_router = 2
        validate_placement(placement, cntrls_per_router, num_rows, num_cols) #nghiant_220525: no longer need a return here
        num_routers = num_rows * num_cols;
        print("[INFO] mesh size: %d x %d" % (num_rows, num_cols))

        # default values for link latency and router latency.
        # Can be over-ridden on a per link/router basis
        link_latency = options.link_latency # used by simple and garnet
        router_latency = options.router_latency # only used by garnet

        # Create the routers in the mesh
        routers = [Router(router_id=i, latency = router_latency) \
            for i in range(num_routers)]
        network.routers = routers

        # link counter to set unique link ids
        link_count = 0

        #nghiant: link nodes to routers
        # Connect each node to the appropriate router
        ext_links = []

        print("[INFO] external linking:")
        mesh_rc_id = 0
        for ni, node in enumerate(nodes):
            if ni < np:
                pi = ni

                entity = placement[pi]
                entity_type = entity["type"]
                entity_id   = entity["id"]
                entity_row  = entity["row"]
                entity_col  = entity["col"]

                router_id = entity_col + entity_row * num_cols

                if buildEnv['PROTOCOL'] == 'Garnet_trace':
                    print(f'link_count: {link_count}  - name: {entity_type}_{entity_id} (true node type: {node.type})- expected: [{entity_row}, {entity_col}] - router_id: {router_id}')
                    ext_links.append(ExtLink(link_id=link_count, ext_node=node,
                                    int_node=routers[router_id],
                                    latency = link_latency))
                    link_count += 1
                else:
                    print("link_count: N/A - name: %s_%d - expected: [%d, %d] - router_id: N/A"
                        % (entity_type, entity_id, entity_row, entity_col))
            else:
                entity_type = entity["type"]
                entity_id   = entity["id"]
                entity_row  = mesh_rc_id // num_cols
                entity_col  = mesh_rc_id % num_cols

                mesh_rc_id += 1

                router_id = entity_col + entity_row * num_cols

                if buildEnv['PROTOCOL'] == 'Garnet_trace':
                    print(f'link_count: {link_count}  - name: Garnet Dest Directory (true node type: {node.type})- expected: [{entity_row}, {entity_col}] - router_id: {router_id}')
                    ext_links.append(ExtLink(link_id=link_count, ext_node=node,
                                    int_node=routers[router_id],
                                    latency = link_latency))
                    link_count += 1
                else:
                    print("link_count: N/A - name: %s_%d - expected: [%d, %d] - router_id: N/A"
                        % (entity_type, entity_id, entity_row, entity_col))


        network.ext_links = ext_links

        #nghiant: link between routers
        # Create the mesh links.
        int_links = []

        # East output to West input links (weight = 1)
        for row in range(num_rows):
            for col in range(num_cols):
                if (col + 1 < num_cols):
                    east_out = col + (row * num_cols)
                    west_in = (col + 1) + (row * num_cols)
                    int_links.append(IntLink(link_id=link_count,
                                             src_node=routers[east_out],
                                             dst_node=routers[west_in],
                                             src_outport="East",
                                             dst_inport="West",
                                             latency = link_latency,
                                             weight=1))
                    link_count += 1

        # West output to East input links (weight = 1)
        for row in range(num_rows):
            for col in range(num_cols):
                if (col + 1 < num_cols):
                    east_in = col + (row * num_cols)
                    west_out = (col + 1) + (row * num_cols)
                    int_links.append(IntLink(link_id=link_count,
                                             src_node=routers[west_out],
                                             dst_node=routers[east_in],
                                             src_outport="West",
                                             dst_inport="East",
                                             latency = link_latency,
                                             weight=1))
                    link_count += 1

        # North output to South input links (weight = 2)
        for col in range(num_cols):
            for row in range(num_rows):
                if (row + 1 < num_rows):
                    north_out = col + (row * num_cols)
                    south_in = col + ((row + 1) * num_cols)
                    int_links.append(IntLink(link_id=link_count,
                                             src_node=routers[north_out],
                                             dst_node=routers[south_in],
                                             src_outport="North",
                                             dst_inport="South",
                                             latency = link_latency,
                                             weight=2))
                    link_count += 1

        # South output to North input links (weight = 2)
        for col in range(num_cols):
            for row in range(num_rows):
                if (row + 1 < num_rows):
                    north_in = col + (row * num_cols)
                    south_out = col + ((row + 1) * num_cols)
                    int_links.append(IntLink(link_id=link_count,
                                             src_node=routers[south_out],
                                             dst_node=routers[north_in],
                                             src_outport="South",
                                             dst_inport="North",
                                             latency = link_latency,
                                             weight=2))
                    link_count += 1


        network.int_links = int_links

    # Register nodes with filesystem
    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node([i],
                    MemorySize(options.mem_size) // options.num_cpus, i)
