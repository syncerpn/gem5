# Copyright (c) 2021 The Regents of the University of California.
# All Rights Reserved
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

from m5.objects import SimpleNetwork, Switch, SimpleExtLink, SimpleIntLink
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

def validate_placement(protocol, placement, max_per_router, spec_row, spec_col):
    #for mesi protocol
    if protocol == 'MESI_Three_Level':
        #nghiant: validate that L0_i and L1_i must be connected to the same router
        for entity_1 in placement:
            if entity_1["type"] == "L0Cache_Controller":
                for entity_2 in placement:
                    if entity_2["type"] == "L1Cache_Controller":
                        if entity_1["id"] == entity_2["id"]:
                            if entity_1["row"] != entity_2["row"] or entity_1["col"] != entity_2["col"]:
                                print("[ERRO] %s_%d and %s_%d must be connected to the same router" % (entity_1["type"], entity_1["id"], entity_2["type"], entity_2["id"]))
                                assert(0)

        #nghiant: validate if exceeding max_per_router; L0s are not counted
        filtered_placement = [entity for entity in placement if entity["type"] != "L0Cache_Controller"]
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

    #for moesi protocol
    elif (protocol == 'MOESI_CMP_directory') or (protocol == 'MESI_Two_Level'):
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
        assert 0, f'[ERRO] protocol not supported: {protocol}; please implement/specify details for it'


class Mesh_nghiant_custom_v2(SimpleNetwork):
    """A simple point-to-point network. This doesn't not use garnet."""

    def __init__(self, ruby_system, placement_file, protocol):
        super().__init__()
        self.netifs = []

        # TODO: These should be in a base class
        # https://gem5.atlassian.net/browse/GEM5-1039
        self.ruby_system = ruby_system
        self._placement_file = placement_file
        self._protocol = protocol

    def connectControllers(self, controllers):
        num_rows, num_cols, placement = parse_placement(self._placement_file)
        nodes = controllers
        
        assert(len(nodes) == len(placement))


        #nghiant: for CMN-600, upto 2 nodes can be connected to a router
        cntrls_per_router = 4
        validate_placement(self._protocol, placement, cntrls_per_router, num_rows, num_cols) #nghiant_220525: no longer need a return here
        num_routers = num_rows * num_cols;
        print("[INFO] mesh size: %d x %d" % (num_rows, num_cols))

        # Create the routers in the mesh
        # routers = [Router(router_id=i, latency = router_latency) for i in range(num_routers)]
        self.routers = [Switch(router_id=i) for i in range(num_routers)]

        # link counter to set unique link ids
        link_count = 0

        #nghiant: link nodes to routers
        # Connect each node to the appropriate router
        ext_links = []
        print("[INFO] external linking:")
        for entity in placement:
            entity_type = entity["type"]
            entity_id   = entity["id"]
            entity_row  = entity["row"]
            entity_col  = entity["col"]
            router_id = entity_col + entity_row * num_cols

            #nghiant:
            #here we assume that controllers in nodes listed in order of their levels L0 L1 L2 Dir
            #for example L0 controller id 0 is always before L0 controller id 1 (strictly needed)
            #and L0 controller is always before L1 (though it is not that necessary in this impl)
            #this is guaranteed during protocol init (please see configs/ruby/MESI_Three_Level.py)
            #nghiant_end
            id_count = -1
            for node in nodes:
                if node.type == entity_type:
                    id_count += 1
                if id_count == entity_id:
                    break

            if id_count != -1:
                print("link_count: %2d  - name: %s_%d - expected: [%d, %d] - router_id: %d"
                    % (link_count, entity_type, entity_id, entity_row, entity_col, router_id))
                ext_links.append(SimpleExtLink(link_id=link_count, ext_node=node, int_node=self.routers[router_id]))
                link_count += 1
            else:
                print("link_count: N/A - name: %s_%d - expected: [%d, %d] - router_id: N/A"
                    % (entity_type, entity_id, entity_row, entity_col))

        #nghiant: link between routers
        # Create the mesh links.
        int_links = []

        # East output to West input links (weight = 1)
        for row in range(num_rows):
            for col in range(num_cols):
                if (col + 1 < num_cols):
                    east_out = col + (row * num_cols)
                    west_in = (col + 1) + (row * num_cols)
                    int_links.append(SimpleIntLink(link_id=link_count, src_node=self.routers[east_out], dst_node=self.routers[west_in]))
                    link_count += 1

        # West output to East input links (weight = 1)
        for row in range(num_rows):
            for col in range(num_cols):
                if (col + 1 < num_cols):
                    east_in = col + (row * num_cols)
                    west_out = (col + 1) + (row * num_cols)
                    int_links.append(SimpleIntLink(link_id=link_count, src_node=self.routers[west_out], dst_node=self.routers[east_in]))
                    link_count += 1

        # North output to South input links (weight = 2)
        for col in range(num_cols):
            for row in range(num_rows):
                if (row + 1 < num_rows):
                    north_out = col + (row * num_cols)
                    south_in = col + ((row + 1) * num_cols)
                    int_links.append(SimpleIntLink(link_id=link_count, src_node=self.routers[north_out], dst_node=self.routers[south_in]))
                    link_count += 1

        # South output to North input links (weight = 2)
        for col in range(num_cols):
            for row in range(num_rows):
                if (row + 1 < num_rows):
                    north_in = col + (row * num_cols)
                    south_out = col + ((row + 1) * num_cols)
                    int_links.append(SimpleIntLink(link_id=link_count, src_node=self.routers[south_out], dst_node=self.routers[north_in]))
                    link_count += 1

        self.ext_links = ext_links
        self.int_links = int_links