""" Full assembly of the parts to form the complete network """

from models.forecastors.Conv3DUNet.unet_utils import *
from torch.autograd import Variable
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels = 5, out_channels = 2 ,bts = 4, bilinear=False,attention = False):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.attention = attention
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.att1 = AttentionBlock(F_g = 512,F_l = 512,n_coefficients = 256)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.att2 = AttentionBlock(F_g = 256,F_l = 256,n_coefficients = 128)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.att3 = AttentionBlock(F_g = 128,F_l = 128,n_coefficients = 64)
        self.up4 = Up(128, 64, bilinear)
        self.att4 = AttentionBlock(F_g = 64,F_l = 64,n_coefficients = 32)
        self.outc = OutConv(64,  self.n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.attention:
            x = self.up1(x5, x4)
            x = self.att1(gate = x,skip_connection = x4)
            x = self.up2(x, x3)
            x = self.att2(gate = x, skip_connection = x3)
            x = self.up3(x, x2)
            x = self.att3(x,x2)
            x = self.up4(x, x1)
            x = self.att4(x,x1)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        logits = self.outc(x)
        return logits