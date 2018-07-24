# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 21:42:20 2018

@author: lenovo
"""


import os 
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch as t

class Spg(data.Dataset):
    def __init__(self,root,train=True,test=False):
        self.test=test
        root_mix=root+'/mix'
        root_vocal=root+'/vocal'
        if test:
            spgs_mix=[os.path.join(root_mix,img) for img in os.listdir(root_mix)]
        else:    
            spgs_vocal=[os.path.join(root_mix,img) for img in os.listdir(root_vocal)]
            spgs_mix=[os.path.join(root_mix,img) for img in os.listdir(root_mix)]
            
        spgs_num=len(spgs_mix)

        if self.test:
            self.spgs_mix=spgs_mix
        elif train:
            self.spgs_mix=spgs_mix[:int(0.7*spgs_num)]
            self.spgs_vocal=spgs_vocal[:int(0.7*spgs_num)]
        else:
            self.spgs_mix=spgs_mix[int(0.7*spgs_num)+1:]
            self.spgs_vocal=spgs_vocal[int(0.7*spgs_num)+1:]
        self.transforms=T.ToTensor()    
    
    def __getitem__(self,index):
        spgs_mix_path=self.spgs_mix[index]
        if self.test:
            label=None
        else:
            spgs_vocal_path=self.spgs_vocal[index]
            label=np.load(spgs_vocal_path)
#            label=self.transforms(label)
            label=t.from_numpy(label)
        data=np.load(spgs_mix_path)
#        data=self.transforms(data)
        data=t.from_numpy(data)
            
        return data,label
    
    def __len__(self):
        return len(self.spgs_mix)