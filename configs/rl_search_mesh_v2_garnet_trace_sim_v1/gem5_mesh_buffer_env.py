import numpy as np
import torch
import os
import time

class gem5_mesh_buffer_env():
    def __init__(self, n_device=6, n_port=8, buffer_dir=None, executer=None, estimator=None, buffer_dumping_interval=1):
        self.performance = {}
        self.buffer_pointer = 0
        self.executer = executer
        self.estimator = estimator
        self.buffer_dir = buffer_dir
        self.buffer_dumping_interval = buffer_dumping_interval

        self.best_state_key = None
        best_perf = 999

        buffer_files = [f for f in os.listdir(buffer_dir) if os.path.isfile(os.path.join(buffer_dir,f))]
        for buffer_file in buffer_files:
            with open(os.path.join(buffer_dir,buffer_file), 'r') as f:
                dataset = np.loadtxt(f, delimiter=',')

            X = dataset[:, 0:-1].astype(np.int32)
            Y = dataset[:,   -1]

            for i in range(X.shape[0]):

                state_key = ','.join([str(x) for x in X[i,:]])
                if state_key in self.performance:
                    print('[WARN] duplicated entries found')
                self.performance[state_key] = Y[i]
                self.buffer_pointer += 1

                if self.best_state_key is None:
                    self.best_state_key = state_key
                elif self.performance[self.best_state_key] > Y[i]:
                    self.best_state_key = state_key

        self.n_device = n_device
        self.n_port = n_port
        print(f'[INFO] buffer contains {len(self.performance)} entries')

    #randomize a state
    def random_state(self) -> torch.Tensor:
        all_ports = list(range(self.n_port))
        np.random.shuffle(all_ports)
        config = np.array(all_ports[:self.n_device]).astype(np.int32)
        state = self._get_obs(config)

        return config, state, perf

    def _get_obs(self, config):
        mesh = np.zeros((1, self.n_device, self.n_port), dtype=np.float32)

        for i in range(self.n_device):
            mesh[0, i, config[i]] = 1

        state = mesh.reshape(1,-1).astype(np.float32)

        return state

    def check_performance(self, config):
        state_key = ','.join([str(x) for x in config])
        if state_key in self.performance:
            print(f'[INFO] use buffer for config: {state_key}')
            perf = self.performance[state_key]
            print(f'[INFO] binary execution time: {perf}')
        else:
            print(f'[INFO] execute config: {state_key}')
            perf = self.executer.execute(config)
            self.performance[state_key] = perf
            print(f'[INFO] executed workload | binary execution time: {perf}')
        
        #feed the trace to estimator
        self.estimator.trace = self.executer.folder_name_from_config(config) + '0/' + 'trace' #TODO: support multiple workloads

        if self.best_state_key is None:
            self.best_state_key = state_key
        elif self.performance[self.best_state_key].sum() > perf.sum():
            self.best_state_key = state_key

        return perf

    def estimate_performance(self, config):
        return self.estimator.execute(config)

    def dump_buffer_to_file(self):
        if (len(self.performance) - self.buffer_pointer >= self.buffer_dumping_interval):
            print(f'[INFO] dumping: buffer has {len(self.performance)} entries, current buffer_pointer: {self.buffer_pointer}')
            file_name = self.buffer_dir + '/' + 'buffer_' + str(int(time.time()))
            with open(file_name, 'a') as f:
                state_keys = list(self.performance.keys())[self.buffer_pointer:len(self.performance)]
                for state_key in state_keys:
                    f.write(','.join([state_key, str(self.performance[state_key])])+'\n')
                    self.buffer_pointer += 1
            print(f'[INFO] dumped: current buffer_pointer: {self.buffer_pointer}')
        else:
            print(f'[INFO] skip dumping due to too few ({len(self.performance) - self.buffer_pointer}) new entries')

    def step(self, old_config, action_d, action_p):
        #mesh config
        new_config = old_config.copy()

        device_a = action_d[0]
        port_b   = action_p[0]

        port_a = old_config[device_a] #device_a's port before swapping

        if port_b in old_config:
            device_b = np.where(new_config == port_b)
            new_config[device_b] = port_a

        new_config[device_a] = port_b

        new_state = self._get_obs(new_config)

        return new_config, new_state