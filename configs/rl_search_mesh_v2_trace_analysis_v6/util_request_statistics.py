import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

### USER SPECIFICATION ###
trace_root_dir = "./"
trace_name = "trace_blackscholes_32" # please rename the trace file such that it ends with _<number_of_devices>
### USER SPECIFICATION ###

def parse_output(file, patterns):
    with open(file) as f:
        data = f.readlines()

    data = [line.strip('\n') for line in data]

    result = []
    for p in patterns:
        result += [line for line in data if re.search(p, line)]

    return result

_DATA_TIMESTAMP = 0
_DATA_PKGID = 1
_DATA_SRCROUTER = 2
_DATA_DSTROUTER = 3
_DATA_VNET = 4

_NODE_TO_NODE_DIST = 1 #cycle

class Progression:
    def __init__(self):
        self.schedule_list = {}
        self.last_key = None
        
    def add_schedule(self, schedule):
        if schedule.start != self.last_key:
            self.schedule_list[schedule.start]  = schedule.duration
            self.last_key = schedule.start
        else:
            if schedule.duration > self.schedule_list[schedule.start]:
                self.schedule_list[schedule.start] = schedule.duration
    
    def last_schedule(self):
        last_start_time = max(self.schedule_list.keys())
        return Schedule(last_start_time, self.schedule_list[last_start_time])
    
    def get_joined_segment(self):
        segment = []
        schedule_start_list = self.schedule_list.keys()
        for schedule_start in schedule_start_list:
            schedule_duration = self.schedule_list[schedule_start]
            if not segment:
                segment.append((schedule_start, schedule_start + schedule_duration))
                continue
            
            last_segment_start = segment[-1][0]
            last_segment_end = segment[-1][1]
            
            if schedule_start <= last_segment_end:
                segment[-1] = (last_segment_start, max(last_segment_end, schedule_start + schedule_duration))
            else:
                segment.append((schedule_start, schedule_start + schedule_duration))
        
        return segment

class Schedule:
    def __init__(self, start=0, duration=0):
        self.start = start
        self.duration = duration
    
    def __repr__(self):
        return f'[INFO] schedule @{self.start} finishes after {self.duration}'

class ScheduleData(Schedule):
    def __init__(self, data, row, col):
        self.start = data[_DATA_TIMESTAMP]
        src_id = get_mesh_index(data[_DATA_SRCROUTER], row, col)
        dst_id = get_mesh_index(data[_DATA_DSTROUTER], row, col)
        d = distance(src_id, dst_id)
        self.duration = 1 + d * _NODE_TO_NODE_DIST

def get_mesh_index(i, row, col):
    return (i % col, i // col)

def distance(a, b):
    xa, ya = a
    xb, yb = b
    return abs(xa - xb) + abs(ya - yb)

NUM_CPUS = int(trace_name.split('_')[-1])

NUM_L2 = 1
NUM_DIR = 1
if NUM_CPUS > 1:
    NUM_L2 = NUM_CPUS // 2
    NUM_DIR = NUM_L2

NUM_ALL = NUM_CPUS + NUM_L2 + NUM_DIR

if NUM_CPUS >= 32:
    matplotlib.rcParams.update({'font.size': 6})

last_trace_content = []

trace_file = f"{trace_root_dir}{trace_name}"
fig_name = f"{trace_file}.svg"

device_name = ["cpu_" + str(i) for i in range(NUM_CPUS)] + ["l2_" + str(i) for i in range(NUM_L2)] + ["dir_" + str(i) for i in range(NUM_DIR)]

with open(trace_file, 'r') as f:
    content = f.readlines()
content = [line.strip('\n') for line in content]

for dli, data_line in enumerate(content):
    items = data_line.split(' ')
    items = [item.strip(': []') for item in items if item.strip(': []')]
    items = items[2:] #remove the first two unused info
    
    data = [None, None, None, None, None]
    data[_DATA_TIMESTAMP] = int(items[0]) // 500
    data[_DATA_PKGID] = int(items[1])
    data[_DATA_SRCROUTER] = int(items[2])
    data[_DATA_DSTROUTER] = int(items[3])
    data[_DATA_VNET] = int(items[4])
    last_trace_content.append(data)

data_transfer_dict = {}
data_transfer_mat = np.zeros((NUM_ALL, NUM_ALL))

for data in last_trace_content:
    src = data[_DATA_SRCROUTER]
    dst = data[_DATA_DSTROUTER]
    k = f"{device_name[src]}.{device_name[dst]}"
    if k not in data_transfer_dict:
        data_transfer_dict[k] = 0
        
    data_transfer_dict[k] += 1
    data_transfer_mat[src, dst] += 1

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
cax = ax.matshow(data_transfer_mat, interpolation='nearest', cmap="hot")
fig.colorbar(cax)

ax.set_xticks(list(range(NUM_ALL)))
ax.set_yticks(list(range(NUM_ALL)))

ax.set_xticklabels(device_name, rotation='vertical')
ax.set_yticklabels(device_name)

plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
plt.grid(which='minor', alpha=0.1)
# plt.show()
plt.title("Data transfer requests: from (row) to (col)", y = -0.1)
plt.savefig(fig_name)