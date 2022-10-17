import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecastors.MobileNet3D.MobNet_utils import * 

class MobileNet(nn.Module):
    def __init__(self,input_channels, num_out_channels ,width_mult=1.):
        super(MobileNet, self).__init__()

        input_channel = input_channels
        last_channel = 1024
        self.num_out_channels = num_out_channels
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # c, n, s
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]
        
        # Encoder Layers
        self.down0 = DoubleConv(input_channel, 32)
        self.down1 = Block(32, int(cfg[0][0]*width_mult), 1)
        self.down2 = Block(int(cfg[0][0]*width_mult), int(cfg[1][0]*width_mult), (2,2,2))
        self.down3 = Block(int(cfg[1][0]*width_mult), int(cfg[2][0]*width_mult), (2,2,2))
        self.down4 = Block(int(cfg[2][0]*width_mult), int(cfg[3][0]*width_mult), (2,2,2))
        self.down5 = Block(int(cfg[3][0]*width_mult), int(cfg[4][0]*width_mult), (2,2,2))
        #Decoder Layers
        self.up1 = Up(int(cfg[4][0]*width_mult),int(cfg[3][0]*width_mult),False)
        self.up2 = Up(int(cfg[3][0]*width_mult),int(cfg[2][0]*width_mult),False)
        self.up3 = Up(int(cfg[2][0]*width_mult),int(cfg[1][0]*width_mult),False)
        self.up4 = Up(int(cfg[1][0]*width_mult),int(cfg[0][0]*width_mult),False)
        self.up5 = Up(int(cfg[0][0]*width_mult),32,False)
        self.scale_conv = OutConv(32,self.num_out_channels)


    def forward(self, x):
        # Encoder Layers
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        # Decoder Layers with residual connections
        x  = self.up1(x5,x4)
        x  = self.up2(x,x3)
        x  = self.up3(x,x2)
        x  = self.up4(x,x1)
        x  = self.up5(x,x0)
        x  = self.scale_conv(x)
        return x

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet(**kwargs)
    return model