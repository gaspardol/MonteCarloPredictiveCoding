from torch import nn
import numpy as np
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.model import bernoulli_fn
from utils.data import get_mnist_data

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def conv_block(in_channels, out_channels, pool=False, pool_no=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              Mish()
              ]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, is_mask=False):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True, pool_no=2)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 256, pool= not is_mask, pool_no=2)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        self.MP = nn.MaxPool2d(2)
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(256,num_classes) if not is_mask else nn.Linear(768, num_classes)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)        
        out = self.res2(out) + out        
        out = self.MP(out) # classifier(out_emb)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        return out  
