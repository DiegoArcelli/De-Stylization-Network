import torch
from torch import nn
# from ibn import Bottleneck_IBN as IBN

class DestylerSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)


    def forward(self, x, y):
        i = 0
        n = len(y)
        for module in self:
            if i < n:
                x = module(x, y[i])
                i += 1
            else:
                x = module(x)
        return x