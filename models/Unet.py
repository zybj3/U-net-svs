# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:54:06 2018

@author: lenovo
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F 
from .BasicModule import BasicModule

class Unet(BasicModule):
    def __init__(self):
        super(Unet,self).__init__()
        self.cat=[]
        
        self.conv1=nn.Sequential(nn.Conv2d(1,16,(5,5),(2,2)),
                                 nn.BatchNorm2d(16),
                                 nn.LeakyReLU(0.2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,(5,5),(2,2)),
                                 nn.BatchNorm2d(32),
                                 nn.LeakyReLU(0.2))
        self.conv3=nn.Sequential(nn.Conv2d(32,64,(5,5),2),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2))
        self.conv4=nn.Sequential(nn.Conv2d(64,128,(5,5),2),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.2))
        self.conv5=nn.Sequential(nn.Conv2d(128,256,(5,5),2),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2))
        self.conv6=nn.Sequential(nn.Conv2d(256,512,(5,5),2),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.2))
        self.deconv1=nn.Sequential(nn.ConvTranspose2d(512,256,(5,5),2),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.deconv2=nn.Sequential(nn.ConvTranspose2d(512,128,(5,5),2),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.deconv3=nn.Sequential(nn.ConvTranspose2d(256,64,(5,5),2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.deconv4=nn.Sequential(nn.ConvTranspose2d(128,32,(5,5),2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.deconv5=nn.Sequential(nn.ConvTranspose2d(64,16,(5,5),2),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        self.final_conv=nn.Sequential(nn.ConvTranspose2d(32,1,5,2),
                                      nn.Sigmoid())
    def forward(self,x):
        
        x1=F.pad(x,(1,2,1,2))
        x1=self.conv1(x1)
#        self.cat.append(x)
        x2=F.pad(x1,(1,2,1,2))
        x2=self.conv2(x2)
#        self.cat.append(x)
        
        x3=F.pad(x2,(1,2,1,2))
        x3=self.conv3(x3)
#        self.cat.append(x)
        x4=F.pad(x3,(1,2,1,2))
        x4=self.conv4(x4)
#        self.cat.append(x)
        x5=F.pad(x4,(1,2,1,2))
        x5=self.conv5(x5)
#        self.cat.append(x)
        x6=F.pad(x5,(1,2,1,2))
        x6=self.conv6(x6)
        
        x6=self.deconv1(x6)
        x6=x6[:,:,1:-2,1:-2]
        x6=F.dropout2d(x6)
        
        x5=t.cat((x6,x5),dim=1)
        x5=self.deconv2(x5)
        x5=x5[:,:,1:-2,1:-2]
        x5=F.dropout2d(x5)
        
        x5=t.cat((x5,x4),dim=1)
        x5=self.deconv3(x5)
        x5=x5[:,:,1:-2,1:-2]
        x5=F.dropout2d(x5)
        
        x5=t.cat((x5,x3),dim=1)
        x5=self.deconv4(x5)
        x5=x5[:,:,1:-2,1:-2]
        
        x5=t.cat((x5,x2),dim=1)
        x5=self.deconv5(x5)
        x5=x5[:,:,1:-2,1:-2]
        
        x5=t.cat((x5,x1),dim=1)
        x5=self.final_conv(x5)
        x5=x5[:,:,1:-2,1:-2]        
        return x5
#        x=F.pad(x,(1,2,1,2))
#        x=self.conv1(x)
#        self.cat.append(x)
#        x=F.pad(x,(1,2,1,2))
#        x=self.conv2(x)
#        self.cat.append(x)
#        x=F.pad(x,(1,2,1,2))
#        x=self.conv3(x)
#        self.cat.append(x)
#        x=F.pad(x,(1,2,1,2))
#        x=self.conv4(x)
#        self.cat.append(x)
#        x=F.pad(x,(1,2,1,2))
#        x=self.conv5(x)
#        self.cat.append(x)
#        x=F.pad(x,(1,2,1,2))
#        x=self.conv6(x)
#        x=self.deconv1(x)
#        x=x[:,:,1:-2,1:-2]
#        x=F.dropout2d(x)
#        x=t.cat((x,self.cat[4]),dim=1)
#        x=self.deconv2(x)
#        x=x[:,:,1:-2,1:-2]
#        x=F.dropout2d(x)
#        x=t.cat((x,self.cat[3]),dim=1)
#        x=self.deconv3(x)
#        x=x[:,:,1:-2,1:-2]
#        x=F.dropout2d(x)
#        x=t.cat((x,self.cat[2]),dim=1)
#        x=self.deconv4(x)
#        x=x[:,:,1:-2,1:-2]
#        x=t.cat((x,self.cat[1]),dim=1)
#        x=self.deconv5(x)
#        x=x[:,:,1:-2,1:-2]
#        x=t.cat((x,self.cat[0]),dim=1)
#        x=self.final_conv(x)
#        x=x[:,:,1:-2,1:-2]        
#        return x



        
