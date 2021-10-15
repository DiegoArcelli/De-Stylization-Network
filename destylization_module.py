import torch
from torch import nn
from torch.nn.modules import flatten
from torch.nn.modules.container import Sequential
from adaptive_instance_normalization import AdaptiveInstanceNormalization
from torch import Tensor
from torchvision import models
from torch.nn import Module
from torchvision.models import densenet

'''
The implementation of the DeStylization module
the paremeter it receives is a list of integer where each integer is 
the number of neurons of one of the last FC layers
'''


class DeStylizationModule(Module):

    def __init__(self):
        super().__init__()
        self.encoder = models.vgg16(pretrained=True).features.eval()
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(25088, 32)
        self.fc_2 = nn.Linear(32, 32)
        self.fc_3 = nn.Linear(32, 32)
        self.fc_4 = nn.Linear(32, 32)

    # load the pretrained vgg model and extract the encoder from it 

    # receives as input an tensor with shape (n, 224, 224, 3)
    # where n is the batch size

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)
        return x
