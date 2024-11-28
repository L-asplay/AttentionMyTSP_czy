from torch.utils.data import Dataset
import torch,math
import os
import pickle
from problems.mec.state_mec import StateMEC
from utils.beam_search import beam_search
import numpy as np

class MEC(object):

    NAME = 'mec'
    Rc = 40.0  # coverage / bias
    UAV_p = 50
    # UAV fly
    height = 10
    g = 9.8  # gravity 
    speed = 20
    quantity_uav = 2
    Cd = 0.3
    A = 0.1
    air_density = 1.225
    P_fly = air_density * A * Cd * pow(speed, 3) / 2 + quantity_uav * g * speed
    P_stay = pow(speed, 3)
    # Iot device energy compute
    switched_capacitance = 1e-7
    v = 4
    # transmit
    B = 1e6
    g0 = 20
    G0 = 5
    upload_P = 3
    noise_P = -90
    hm = 0
    d_2 = pow(Rc, 2) + pow(height, 2)
    upload_speed = B * math.log2(1 + g0 * G0 * upload_P / pow(noise_P, 2) / (pow(hm, 2) + d_2))

    @staticmethod
    def get_costs(dataset, pi):
       
        # Gather dataset in order of tour
        assert 0, "to be realized"
        return None,None# 总能耗

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MECDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMEC.initialize(*args, **kwargs)

    @staticmethod
    def beam_search():
        # TODO: complete this function
        pass

def make_instance(args):
    task_data, UAV_start_pos, task_position,CPU_circles, IoT_resource, UAV_resource,time_window, *args = args
    return {
        'task_data' : torch.tensor(task_data, dtype=torch.float),
        'UAV_start_pos' : torch.tensor(UAV_start_pos, dtype=torch.float),
        'task_position' : torch.tensor(task_position, dtype=torch.float),
        'CPU_circles' : torch.tensor(CPU_circles, dtype=torch.float),
        'IoT_resource' : torch.tensor(IoT_resource, dtype=torch.float),
        'UAV_resource': torch.tensor(UAV_resource, dtype=torch.float),
        'time_window' : torch.tensor(time_window, dtype=torch.float),
    }

class MECDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, dependency=[],distribution=None):
        super(MECDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
                task_data = np.random.uniform(size=(num_samples, size, 1), low=0, high=1000)
                UAV_start_pos = np.random.randint(size=(num_samples, 1, 2), low = 0, high = 500)
                task_position = np.random.uniform(size=(num_samples, size, 2), low=0, high=500)
                CPU_circles = np.random.randint(size=(num_samples, size, 1), low=0, high=1000)
                IoT_resource = np.random.randint(size=(num_samples, size, 1), low=100, high=200)
                UAV_resource = np.max(CPU_circles, axis=1, keepdims=True) // 4

                time_window = np.random.randint(size=(num_samples, size, 2), low=0, high=100)
                time_window = np.sort(time_window, axis=2)
                dep_window = np.take(time_window, indices=dependency, axis=1)
                dep_window = np.sort(dep_window.reshape(num_samples, -1), axis=1).reshape(num_samples, len(dependency), 2)
                np.put_along_axis(time_window, np.array(dependency)[None, :, None].astype(int), dep_window, axis=1)
        
                data = list(zip(
                        task_data.tolist(),
                        UAV_start_pos.tolist(),
                        task_position.tolist(),
                        CPU_circles.tolist(),
                        IoT_resource.tolist(),
                        UAV_resource.tolist(),
                        time_window.tolist()
                        ))


        self.data = self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
