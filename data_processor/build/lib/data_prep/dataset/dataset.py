import matplotlib.pyplot as plt
from netCDF4 import Dataset
from skimage import filters
import torch.nn as nn
import netCDF4 as nc
import pandas as pd
import numpy as np
import random
import torch
import os
from skimage import filters


DATAPATH = os.environ.get("DATAPATH","/home/resifis/Desktop/kaustcode/Packages/processed_clean_data")



class Dataset2D():
    def __init__(self,solar_type,data_type = "Train"):
        self.solar_type  = "DNI"
        self.hcloud = Dataset2D._read_data("hcloud")
        self.mcloud = Dataset2D._read_data("mcloud")
        self.lcloud = Dataset2D._read_data("lcloud")
        self.water_vapor = Dataset2D._read_data("water_vapor")
        self.ozone = Dataset2D._read_data("ozone")
        self.aerosol = Dataset2D._read_data("aerosol")
        self.len = 48222
        self.solar_type = solar_type
        self.target_data = Dataset2D._read_target(self.solar_type)
        self.extra_input = Dataset2D._read_target("DNI")
        self.output = dict()
        
    
    @staticmethod
    def _read_data(data_type):
        if data_type == "hcloud":
            hcloud = Dataset(os.path.join(DATAPATH,"hcloud.nc"))
            return hcloud
        elif data_type == "mcloud":
            mcloud = Dataset(os.path.join(DATAPATH,"mcloud.nc"))
            return mcloud
        elif data_type == "lcloud":
            lcloud = Dataset(os.path.join(DATAPATH,"lcloud.nc"))
            return lcloud
        elif data_type == "water_vapor":
            water_vapor = Dataset(os.path.join(DATAPATH,"water_vapor_new.nc"))
            return water_vapor
        elif data_type == "ozone":
            ozone = Dataset(os.path.join(DATAPATH,"ozone.nc"))
            return ozone
        else:
            aerosol = Dataset(os.path.join(DATAPATH,"aod.nc"))
            return aerosol
    @staticmethod
    def _read_target(target_type):
        if target_type == "GHI":
            GHI = Dataset(os.path.join(DATAPATH,"ghi.nc"))
            return GHI
        elif target_type == "DHI":
            DHI = Dataset(os.path.join(DATAPATH,"dhi.nc"))
            return DHI
        else :
            DNI = Dataset(os.path.join(DATAPATH,"dni.nc"))
            return DNI
        
        
    @staticmethod
    def _filtering(list_data):
        data = list_data.copy()
        list_tensors = []
        for i in range(len(data)):
            arr = filters.sobel(np.array(data[i]))
            list_tensors.append(torch.tensor(arr,dtype = torch.float))
        return list_tensors
    
    def _get_tensors(self,item):
        all_data = []
        data_hcloud =  torch.tensor(self.hcloud.variables["cc"][item,:,:],dtype = torch.float)
        data_mcloud =  torch.tensor(self.mcloud.variables["cc"][item,:,:],dtype = torch.float)
        data_lcloud =  torch.tensor(self.lcloud.variables["cc"][item,:,:],dtype = torch.float)
        dni = torch.tensor(self.extra_input.variables['dni'][item],dtype = torch.float)
        data_aerosol = torch.tensor(self.aerosol.variables["aod5503d"][item,:,:],dtype = torch.float)
        data_ozone =   torch.tensor(self.ozone.variables["o3rad"][item,:,:],dtype = torch.float)
        data_water_vapor = torch.tensor(self.water_vapor.variables["qvapor"][item,:,:],dtype = torch.float)
        all_data.extend([torch.flip(data_hcloud,dims = [0]).unsqueeze(0),
                         torch.flip(data_mcloud,dims = [0]).unsqueeze(0),
                         torch.flip(data_lcloud,dims = [0]).unsqueeze(0),
                         torch.flip(dni,dims = [0]).unsqueeze(0),
                         torch.flip(data_aerosol,dims = [0]).unsqueeze(0),
                         torch.flip(data_ozone,dims = [0]).unsqueeze(0),
                         torch.flip(data_water_vapor,dims = [0]).unsqueeze(0),
                        ])
        
        filtered = Dataset2D._filtering(all_data)
        all_data.extend(filtered)
        target = torch.tensor(self.target_data.variables[self.solar_type.lower()][item],dtype = torch.float)
        target = torch.flip(target,dims = [0])
        return all_data,target
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self,item):
        out = dict()
        all_data,target = self._get_tensors(item)
        out['data'] = torch.cat(all_data,dim = 0)
        out['target'] = target
        
        return out
        