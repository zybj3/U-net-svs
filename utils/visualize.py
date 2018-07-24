# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:16:57 2018

@author: lenovo
"""

import visdom
import numpy as np
import time

class Visualizer(object):
    def __init__(self,env='default',**kwargs):
        self.vis=visdom.Visdom(env=env,**kwargs)
        self.index={}
        self.log_txt=''
        
    def reinit(self,env='default',**kwargs):
        self.vis=visdom.Visdom(env=env,**kwargs)
        return self
    
    def plot(self,name,y,**kwargs):
        x=self.index.get(name,0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update= 0 if x==0 else 'append',
                      **kwargs)
        self.index[name] =x+1
        
    def img(self,name,img_,**kwargs):
        self.vis.image(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs)
        
    def log(self,info,win='log_txt'):
        self.log_txt += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'),
                         info=info))
        self.vis.text(self.log_txt,win=win)
        
    def __getattr__(self,name):
        return getattr(self.vis,name)
    
    def plot_many(self,d):
        for k,v in d.items():
            self.plot(k,v)
            
    def img_many(self,d):
        for k,v in d.items():
            self.img(k,v)