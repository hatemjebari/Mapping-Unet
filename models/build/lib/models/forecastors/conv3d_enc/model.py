import torch
import numpy as np 
import pandas as pd
import torch.nn as nn



class EncoderLayer(nn.Module):
    def __init__(self,params_encoder,init_weights = True):
        super(EncoderLayer,self).__init__()
        self.params = params_encoder
        self.init_weights = init_weights
        self.conv1 = nn.Conv3d(in_channels = self.params["in_channels"][0],
                               out_channels = self.params["out_channels"][0],
                               kernel_size = self.params["kernel_sizes"][0])
        self.batchnorm1 = nn.BatchNorm3d(num_features =self.params["out_channels"][0])
        self.conv2 = nn.Conv3d(in_channels = self.params["in_channels"][1],
                               out_channels = self.params["out_channels"][1],
                               kernel_size = self.params["kernel_sizes"][1])
        self.batchnorm2 = nn.BatchNorm3d(num_features =self.params["out_channels"][1])
        self.conv3 = nn.Conv3d(in_channels = self.params["in_channels"][2],
                               out_channels = self.params["out_channels"][2],
                               kernel_size = self.params["kernel_sizes"][2])
        self.batchnorm3 = nn.BatchNorm3d(num_features =self.params["out_channels"][2])
        self.activation = nn.ReLU(True)
        
        self.maxpool3d = nn.MaxPool3d(kernel_size = 2)
        if self.init_weights:
            self._initialize_weights()
    
                
    def forward(self,x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.activation(out)
        out = self.maxpool3d(out)
        residual1 = out 
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.activation(out)
        out = self.maxpool3d(out)
        residual2 = out
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.activation(out)
        return out,residual1,residual2
    
    
class DecoderLayer(nn.Module):
    def __init__(self,params_decoder,init_weights = True):
        super(DecoderLayer,self).__init__()
        self.params = params_decoder
        self.init_weights = init_weights
        #build block1
        self.upsample_layer1 = nn.Upsample(size = self.params["upsampling"][0])
        self.deconv1 = nn.ConvTranspose3d(in_channels = self.params["in_channels"][0], 
                                           out_channels = self.params["out_channels"][0], 
                                           kernel_size = self.params["kernel_sizes"][0])
        self.batchnorm1 = nn.BatchNorm3d(num_features = self.params["out_channels"][0])
        #build block1
        self.upsample_layer2 = nn.Upsample(size = self.params["upsampling"][1])
        self.deconv2 = nn.ConvTranspose3d(in_channels = self.params["in_channels"][1], 
                                           out_channels = self.params["out_channels"][1], 
                                           kernel_size = self.params["kernel_sizes"][1])
        self.batchnorm2 = nn.BatchNorm3d(num_features = self.params["out_channels"][1])
        #build block1
        self.upsample_layer3 = nn.Upsample(size = self.params["upsampling"][2])
        self.deconv3 = nn.ConvTranspose3d(in_channels = self.params["in_channels"][2], 
                                           out_channels = self.params["out_channels"][2], 
                                           kernel_size = self.params["kernel_sizes"][2])
        self.batchnorm3 = nn.BatchNorm3d(num_features = self.params["out_channels"][2])
        self.activation = nn.ReLU(True)
        self.residual_upsampling = nn.Upsample(size = (9, 31, 31))
        if self.init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x,residual1,residual2):
        #first block
        out = self.upsample_layer1(x)
        out = self.deconv1(out)
        out = self.batchnorm1(out)
        out = self.activation(out)
        #second block
        out = self.upsample_layer2(out)
        out+= residual2
        out = self.deconv2(out)
        out = self.batchnorm2(out)
        out = self.activation(out)
        #Third block
        
        out = self.residual_upsampling(out)
        out += residual1
        out = self.upsample_layer3(out)
        out = self.deconv3(out)
        out = self.batchnorm3(out)
        out = self.activation(out)
        return out
    
    
class MeteoModel(nn.Module):
    def __init__(self,params_model):
        super(MeteoModel,self).__init__()
        self.params_model = params_model
        self.params_encoder = self.params_model["params_encoder"]
        self.params_decoder = self.params_model["params_decoder"]
        self.encoder = EncoderLayer(self.params_encoder)
        self.decoder = DecoderLayer(self.params_decoder)
    def forward(self,x):
        out,residual1,residual2 = self.encoder(x)
        out = self.decoder(out,residual1,residual2)
        return out