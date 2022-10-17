""" Full assembly of the parts to form the complete network """
import torch.nn as nn
from models.forecastors.mulit_encoder_unet.unet_utils import *

class UNet(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 2 , bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(self.n_channels, 10)
        self.down1 = Down(120, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 240 // factor, bilinear)
        self.up4 = Up(240, 120//factor, bilinear)
        self.outc = OutConv(120,  self.n_classes)
        
    
    def encode_all(self,data_dict):
        list_enc = []
        for key in data_dict.keys():
            if key != "target":
                list_enc.append(self.inc(data_dict[key]))
        x = torch.cat(list_enc,dim = 1)
        return x
            
    def forward(self, data_dict):
        x1 = self.encode_all(data_dict)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits