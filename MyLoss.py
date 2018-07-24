# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 20:57:07 2018

@author: lenovo
"""

import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, x,predict,y):
        return  torch.sum(torch.abs(x*predict-y))