import pandas as pd
import numpy as np
import torch
from netCDF4 import Dataset
import netCDF4 as nc
import os



class MethoDataSets():
    def __init__(self,nc_path:str,
                 config
                ):
        """This class will read an Netcdf data
        create tensors datasets with corresponding targets
        nc_path: path for NetCdf data
        window_size: the time window for the data
        target_len: the number of timesteps that we are going to forecast
        """
        self.config = config
        self.nc_path = nc_path
        self.window_size = self.config.window_size
        self.target_len = self.config.target_len
        self.netdataset = Dataset(self.nc_path, mode='r')
        self.all_data = torch.tensor(self.netdataset.variables["feature"],dtype = torch.float)
        self.metodata,self.metotarget = self._windo_adder(self.all_data,self.window_size,self.target_len)
        self.out = dict()
        self.out["data"] = self.metodata
        self.out["target"] = self.metotarget
        
    def _windo_adder(self,data,window_size,target_len):
        self.j = 0 
        while self.j <= data.shape[1]-22:
            dataset = data[:, self.j: self.j+window_size,:,:].unsqueeze(0)
            target = data[0, self.j: self.j+window_size,:,:].unsqueeze(0)
            if self.j == 0 :
                set_ = dataset
                set_target = target
            else :
                all_target = set_target
                all_data = set_
                all_data = torch.cat([all_data,dataset],dim = 0)
                all_target = torch.cat([all_target,target],dim = 0)
                set_ = all_data
                set_target = all_target
            self.j +=2
        return all_data,all_target
    
    def __len__(self):
        return int(self.j/2) - 1
    
    def __getitem__(self,item):
        self.item_dict = dict()
        self.item_dict["data"] = self.out["data"][item,:,:,:,:]
        self.item_dict["target"] = self.out["target"][item,:,:,:,:]
        return self.item_dict