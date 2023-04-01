import torch
import copy
import numpy as np
class HardBuffer:
    def __init__(self, batch_size=64, buffer_size=256):
        self.data = None
        self.label = None
        self.full = False
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def is_full(self):
        return self.full

    def put(self, data, label):
        if self.data is None:
            self.data = copy.deepcopy(data)
            self.label = copy.deepcopy(label)
        else:
            self.data = torch.vstack((self.data, copy.deepcopy(data)))
            self.label = torch.cat((self.label, copy.deepcopy(label)), dim=0)
        if self.data.shape[0] == self.buffer_size:
            self.full = True

    def update(self, data, label):
        a = [i for i in range(self.buffer_size)]
        replace_size = data.shape[0]
        idx = torch.LongTensor(np.random.choice(self.buffer_size, replace_size, replace=False)).cuda()
        self.data.index_copy_(0, idx, data)
        self.label.index_copy_(0, idx, label)
    
    def sample(self):
        idx = torch.LongTensor(np.random.choice(self.data.shape[0], self.batch_size, replace=False)).cuda()
        sample_data = self.data.index_select(0, idx)
        sample_label = self.label.index_select(0, idx)
        return (sample_data, sample_label)

