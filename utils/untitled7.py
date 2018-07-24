# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:11:26 2018

@author: lenovo
"""

from torch import nn

class NetG(nn.Module):
    def __init__(self,opt):
        self.ngf=opt.ngf
        self.main=nn.Sequential(
                nn.ConvTranspose2d(opt.nz,8*self.ngf,4,1,0,bias=False),
                nn.BatchNorm2d(8*self.ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(8*self.ngf,4*self.ngf,4,2,1,bias=False),
                nn.BatchNorm2d(4*self.ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(4*self.ngf,2*self.ngf,4,2,1,bias=False),
                nn.BatchNorm2d(2*self.ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(2*self.ngf,2*self.ngf,4,2,1,bias=False),
                nn.BatchNorm2d(2*self.ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(2*self.ngf,self.ngf,4,2,1,bias=False),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(self.ngf,3,5,3,1,bias=False),
                nn.Tanh()
                )
    def foward(self,input):
        return self.main(input)

class NetD(nn.Module):
    def __init__(self,opt):
        super(NetD,self).__init__()
        ndf=opt.ndf
        self.main=nn.Sequential(
                nn.Conv2d(3,ndf,5,3,1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
                nn.BatchNorm2d(ndf*4,),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(ndf*8,1,4,1,0,bias=False),
                nn.Sigmoid())
    def forward(self,input):
        return self.main(input).view(-1)