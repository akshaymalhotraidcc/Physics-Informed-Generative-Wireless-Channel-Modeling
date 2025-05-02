import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

def barrier_function(upper_bound,inp):

    barrier_relu = nn.ReLU()
    mask = (upper_bound>inp)*1.0
    nb_mask = (upper_bound<inp)*1.0
    barrier_loss = barrier_relu(-torch.log(upper_bound-inp))
    print(mask)
    return torch.mean(barrier_loss)

def gen_sigmoid(upper_bound,inp):

    beta = 100.0
    temp = 1.0
    sigmoid = beta/(1+torch.exp(temp*(upper_bound-inp-1)))
    
    return torch.mean(sigmoid)

def relu_barrier(upper_bound,inp):

    barrier_relu = nn.ReLU()
    beta = 1e5
    relu = beta*barrier_relu(inp-upper_bound)
    
    return torch.mean(relu)