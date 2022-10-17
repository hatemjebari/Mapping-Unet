from models.forecastors.ResNet3D.res_utils import *
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, in_channels,bilinear = False):
        super().__init__()
        #Resnet18 Encoder
        self.layer0 = DoubleConv(in_channels,32)
        

        self.layer1 = nn.Sequential(
            ResBlock(32, 64, downsample=True),
            ResBlock(64, 64, downsample=False)
        )

        
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False)
        )
        self.layer5 = nn.Sequential(
            ResBlock(512, 1024, downsample=True),
            ResBlock(1024, 1024, downsample=False)
        )
        factor = 1 
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)
        self.out = OutConv(32,1)

    def forward(self, x):
        #Resnet Encoder
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        # Decoder: Reconstraction
        x = self.up1(x6,x5)
        x = self.up2(x,x4)
        x = self.up3(x,x3)
        x = self.up4(x,x2)
        x = self.up5(x,x1)
        x = self.out(x)
        return x