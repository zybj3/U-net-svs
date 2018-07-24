# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:54:06 2018

@author: lenovo
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F 
import torchvision as tv
from PIL import ImageShow

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.cat=[]
        self.conv1=nn.Sequential(nn.Conv2d(1,16,(5,5),2),
                                 nn.BatchNorm2d(16),
                                 nn.LeakyReLU(0.2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,(5,5),2),
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
        x=F.pad(x,(1,2,1,2))
        x=self.conv1(x)
        self.cat.append(x)
        x=F.pad(x,(1,2,1,2))
        x=self.conv2(x)
        self.cat.append(x)
        x=F.pad(x,(1,2,1,2))
        x=self.conv3(x)
        self.cat.append(x)
        x=F.pad(x,(1,2,1,2))
        x=self.conv4(x)
        self.cat.append(x)
        x=F.pad(x,(1,2,1,2))
        x=self.conv5(x)
        self.cat.append(x)
        x=F.pad(x,(1,2,1,2))
        x=self.conv6(x)
        x=self.deconv1(x)
        x=x[:,:,1:-2,1:-2]
        x=F.dropout2d(x)
        x=t.cat((x,self.cat[4]),dim=1)
        x=self.deconv2(x)
        x=x[:,:,1:-2,1:-2]
        x=F.dropout2d(x)
        x=t.cat((x,self.cat[3]),dim=1)
        x=self.deconv3(x)
        x=x[:,:,1:-2,1:-2]
        x=F.dropout2d(x)
        x=t.cat((x,self.cat[2]),dim=1)
        x=self.deconv4(x)
        x=x[:,:,1:-2,1:-2]
        x=t.cat((x,self.cat[1]),dim=1)
        x=self.deconv5(x)
        x=x[:,:,1:-2,1:-2]
        x=t.cat((x,self.cat[0]),dim=1)
        x=self.final_conv(x)
        x=x[:,:,1:-2,1:-2]
#        
        return x
        
        
if __name__=="__main__":
    unet=Unet()
    x=t.randn(1,1,512,128)
    x=unet(x)
    print(x.size())
    x=tv.transforms.ToPILImage()(x[0])
    x.show()
    
