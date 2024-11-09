from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
       
        # Check that tours are valid, i.e. contain 0 to n -1
       
        # Sorting it should give all zeros at front and then 1...n

        # Visiting depot resets capacity so we add demand = -capacity 

        # check the constraint of time window
       
        return # 总能耗

    @staticmethod
    def make_dataset(*args, **kwargs):
        return ATTDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return ATTDataset.initialize(*args, **kwargs)

    @staticmethod
    def beam_search():
        # TODO: complete this function
        pass


class ATTDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(ATTDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            
            data = [] # a sample

        self.data = data
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
