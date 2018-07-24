# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:17:48 2018

@author: lenovo
"""

import os 
import torch as t
import models
from dataset.dataset import Spg
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
from MyLoss import MyLoss
import time

import numpy as np
from librosa.core import istft, load, stft, magphase
from librosa.output import write_wav


def train():
    vis=Visualizer(env='svs')  
    model=getattr(models,'Unet')()
    model.train().cuda()
      
    train_data=Spg('F:/crop_test',train=True)
    val_data=Spg('F:/crop_test',train=False)
    train_dataloader=DataLoader(train_data,batch_size=4,drop_last=True)
    val_dataloader=DataLoader(val_data,batch_size=1,drop_last=True)
    loss_meter = meter.AverageValueMeter()
    lr=0.001
    lr_decay=0.05

    optimizer=t.optim.Adam(model.parameters(),lr=lr,weight_decay=lr_decay)
    previous_loss=1e100
    
    for epoch in range(5):
        loss_meter.reset()
        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            
            input1=Variable(data).cuda()
            target=Variable(label).cuda()
            optimizer.zero_grad()
            scroe=model(input1)
            loss=MyLoss()(input1,scroe,target).cuda()
            loss.backward()
            optimizer.step()
            
            
            
            loss_meter.add(loss.data.item())
            
            if ii%20==19:
                vis.plot('loss',loss_meter.value().item())
        prefix='G:/Unet_svs/check/'
        name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        
        t.save(model.state_dict(),name)
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        

        previous_loss = loss_meter.value()[0]
                

def test():
    vis=Visualizer(env='svs')  
    model=getattr(models,'Unet')().eval()
#    model.cuda()
    model.load_state_dict(t.load('G:/Unet_svs/check/epoch_219__0724_16_57_35.pth'))
    mix_wav, _ = load("C:/Users/lenovo/Music/c.mp3", sr=8192)
    mix_wav_mag, mix_wav_phase = magphase(stft(mix_wav, n_fft=1024, hop_length=768))
    START = 700
    END = START + 128
    
    mix_wav_mag=mix_wav_mag[:, START:END]
    mix_wav_phase=mix_wav_phase[:, START:END]
    
    print(mix_wav_mag.shape)
    
    gg=mix_wav_mag[1:]
    gg=t.from_numpy(gg)
    gg.unsqueeze_(0)
    gg.unsqueeze_(0)
    vis.img('a',gg)
    print(gg.shape)
    with t.no_grad():
        gg=Variable(gg)
    score=model(gg)
    predict=gg.data*score.data
    print(predict.shape)
    target_pred_mag=predict.view(512,128).cpu().numpy()
    target_pred_mag = np.vstack((np.zeros((128)), target_pred_mag ))
    vis.img('b',t.from_numpy(target_pred_mag))
    print(target_pred_mag.shape)
    write_wav(f'C:/Users/lenovo/Music/pred_vocal.wav', istft(
    target_pred_mag*mix_wav_phase
#     (mix_wav_mag * target_pred_mag) * mix_wav_phase
    , win_length=1024, hop_length=768), 8192, norm=True)
    write_wav(f'C:/Users/lenovo/Music/pred_mix.wav', istft(
    mix_wav_mag * mix_wav_phase
    , win_length=1024, hop_length=768), 8192, norm=True)

#    write_wav(f'C:/Users/lenovo/Music/pred_mix111.wav', istft(
#    mix_wav_mag , win_length=1024, hop_length=768), 8192, norm=True)
    
    
    



if __name__=='__main__':
    test()
    
   
    