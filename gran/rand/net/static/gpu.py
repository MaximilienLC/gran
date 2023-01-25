# separate1d = 
# https://discuss.pytorch.org/t/separate-the-linear-layer-for-different-features/53252/24

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, cfg):

        self.layer = torch.nn.Conv1d(1, 1, stride=kernel_size=1, groups=3)