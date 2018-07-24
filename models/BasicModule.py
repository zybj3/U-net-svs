# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 22:26:05 2018

@author: lenovo
"""

import time
import torch as t

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))
        
    def load(self,path):
        self.load_state_dict(t.load(path))
        
    def save(self,name=None):
        if name is None:
            prefix='G:/chapter6/checkpoints/'+self.model_name+'_'
            name=time.strftime(prefix +'%m%d_%H%M%S.pth')
        t.save(self.state_dict(),name)
        return name
    
    
class Flat(t.nn.Module):
    def __init__(self):
        super(Flat,self).__init__()
        
    def forward(self,x):
        return x.view(x.size()[0],-1)