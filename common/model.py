import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, keypoints_num=17, n=1, dropout=0.5):
        super().__init__()
        self.keypoints_num = keypoints_num
        self.input_shape = self.keypoints_num * 4 * 2
        self.output_shape = self.keypoints_num * 3
        self.faltten = nn.Flatten()
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.w1 = nn.Linear(self.input_shape, 1024 * n)
        self.w2 = nn.Linear(1024 * n, 2048 * n)
        self.w3 = nn.Linear(2048* n, 2048 * n)
        self.w4 = nn.Linear(2048 * n, 1024 * n)
        self.w5 = nn.Linear(1024 * n, self.output_shape)

        self.norm1 = nn.BatchNorm1d(1024 * n)
        self.norm2 = nn.BatchNorm1d(2048 * n)
        self.norm3 = nn.BatchNorm1d(2048 * n)
        self.norm4 = nn.BatchNorm1d(2048 * n)
        self.norm5 = nn.BatchNorm1d(1024 * n)
        
    def forward(self, x):
        x = self.faltten(x)
        x1 = self.w1(x)
        x1 = self.norm1(x1)
        
        x2 = self.w2(x1)
        x2 = self.norm2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        x2 = self.w3(x2)
        x2 = self.norm3(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.w3(x2)
        x2 = self.norm4(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        x2 = self.w4(x2)
        x2 = self.norm5(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        y = x1 + x2
        y = self.w5(y)
        out = y.view(-1, self.keypoints_num, 3)
        
        return out