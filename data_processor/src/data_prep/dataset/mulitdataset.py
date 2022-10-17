from scipy.ndimage import uniform_filter1d
from netCDF4 import Dataset
from skimage import filters
import torch.nn as nn
import netCDF4 as nc
import pandas as pd
import numpy as np
import random
import torch
import os

DATAPATH = os.environ.get("DATAPATH","/home/resifis/Desktop/kaustcode/Packages/processed_clean_data")


class WRFDataset():
    """ This Class is an iterable Dataset class which will be needed by the dataloader
        to get the batch data"""
    def __init__(
        self,
        window_size: int = 10,
        solar_type: str = "GHI",
        data_type: str = "train"
    ):
        self.hcloud = WRFDataset._read_data("hcloud")
        self.mcloud = WRFDataset._read_data("mcloud")
        self.lcloud = WRFDataset._read_data("lcloud")
        self.water_vapor = WRFDataset._read_data("water_vapor")
        self.ozone = WRFDataset._read_data("ozone")
        self.aerosol = WRFDataset._read_data("aod")
        
        self.t2 = WRFDataset._read_data("t2")
        self.td2 = WRFDataset._read_data("td2")
        self.mslp = WRFDataset._read_data("mslp")
        self.rain = WRFDataset._read_data("rain")
        self.ws = WRFDataset._read_data("WS")
        self.wd = WRFDataset._read_data("WD")
        
        self.solar_type = solar_type
        self.target_data = WRFDataset._read_target(self.solar_type)
        self.window_size = window_size
        self.data_type = data_type
        self.idx_row = [i for i in range(213) if i%2 == 0]
        self.idx_col = [i for i in range(288) if i%2 == 0]
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
        elif data_type == "aod":
            aerosol = Dataset(os.path.join(DATAPATH,"aod.nc"))
            return aerosol
        elif data_type == 't2':
            t2 = Dataset(os.path.join(DATAPATH,"t2.nc"))
            return t2
        elif data_type == 'td2':
            td2 = Dataset(os.path.join(DATAPATH,"dt2.nc"))
            return td2
        elif data_type == "mslp":
            mslp = Dataset(os.path.join(DATAPATH,"mslp.nc"))
            return mslp
        elif data_type == 'WS':
            ws = Dataset(os.path.join(DATAPATH,"WS.nc"))
            return ws
        elif data_type == "WD":
            wd = Dataset(os.path.join(DATAPATH,"WD.nc"))
            return wd 
        elif data_type == "rain":
            rain  = Dataset(os.path.join(DATAPATH,"rain.nc"))
            return rain
        
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
    def _normalize(tensor):
        normalized_tensor = nn.functional.normalize(tensor,dim = 0)
        return tensor
    
    
    @staticmethod
    def _filtering(data):
        arr_data = data.copy()
        list_tensors = []
        for i in range(arr_data.shape[0]):
            arr = filters.sobel(np.flip(arr_data[i],axis = 0))
            list_tensors.append(torch.tensor(arr,dtype = torch.float).unsqueeze(0))
        filtered_data = torch.cat(list_tensors,dim = 0)
        filtered_data = filtered_data.unsqueeze(0)
        return filtered_data
    
    def _scaling(self,reshaped_data):
        scaled_data = []
        for data in reshaped_data:
            list_steps = []
            for step in range(self.window_size):
                step_data = torch.tensor(data[step])
                max_ = step_data.max()
                list_steps.append((step_data/(max_+ 1e-5)).unsqueeze(0))
            steps_torch = torch.cat(list_steps,dim = 0)
            scaled_data.append(np.array(steps_torch))
        return scaled_data
            
                
            
        
    def _get_tensor_data(self,item):
        raw_data = list()
        reshaped_data = list()
        all_data = list()
        enc_data = dict()
        data_hcloud =  1e-1 * np.array(self.hcloud.variables["cc"][item:item+self.window_size,:,:])
        data_mcloud =  1e-1*np.array(self.mcloud.variables["cc"][item:item+self.window_size,:,:])
        data_lcloud =  1e-1*np.array(self.lcloud.variables["cc"][item:item+self.window_size,:,:])
        data_aerosol = 100*np.array(self.aerosol.variables["aod5503d"][item:item+self.window_size,:,:])
        data_ozone =   1e7*np.array(self.ozone.variables["o3rad"][item:item+self.window_size,:,:])
        data_water_vapor = 1e3*np.array(self.water_vapor.variables["qvapor"][item:item+self.window_size,:,:])
        t2 = 1e-1*np.array(self.t2.variables['t2'][item:item+self.window_size,:,:])
        td2 = 1e-1*np.array(self.td2.variables['td2'][item:item+self.window_size,:,:])
        mslp = 1e-3 * np.array(self.mslp.variables['mslp'][item:item+self.window_size,:,:])
        rain = 1e5 * np.array(self.rain.variables['rain'][item:item+self.window_size,:,:])
        ws = np.array(self.ws.variables['WS'][item:item+self.window_size,:,:])
        wd = np.array(self.wd.variables['WD'][item:item+self.window_size,:,:])
        raw_data.extend([data_hcloud,
                         data_mcloud,
                         data_lcloud,
                         data_aerosol,
                         data_ozone,
                         data_water_vapor,
                         t2,
                         td2,
                         mslp,
                         rain,
                         ws,
                         wd,
                         ])
        
        for data in raw_data : 
            shrink_data = np.delete(data,self.idx_row,axis = 1)
            shrink_data = np.delete(shrink_data,self.idx_col,axis = 2)
            reshaped_data.append(shrink_data)
            
        reshaped_scaled_data = self._scaling(reshaped_data)
        
        
                
        for i,data in enumerate(reshaped_scaled_data):
            list_input_feat = []
            list_input_feat.append(WRFDataset._normalize(torch.flip(torch.tensor(data),dims = [1])).unsqueeze(0))
            list_input_feat.append(WRFDataset._filtering(data))
            
            enc_data[f"feat_{i}"] = torch.cat(list_input_feat,dim = 0)
    
        
        target_cdf = np.array(self.target_data.variables[self.solar_type.lower()][item:item+self.window_size,:,:])
        target_cdf = np.delete(target_cdf,self.idx_row,axis = 1)
        target_cdf = np.delete(target_cdf,self.idx_col,axis = 2)
        target_cdf = torch.tensor(target_cdf)
        target = target_cdf.squeeze(1)
        target = torch.flip(target,dims = [1])
        
        enc_data["target"] = target
        
        return enc_data
        
    def __len__(self):
        full_len = self.hcloud.variables["cc"].shape[0]
        if self.data_type == "train":
            data_len = full_len - int(0.1*full_len)
        else:
            data_len = int(0.1*full_len)
        return data_len  - self.window_size
    
    def __getitem__(self,item):
        if self.data_type == "valid":
            item = item + int(0.9*self.hcloud.variables["cc"].shape[0])
        self.output= self._get_tensor_data(item)
        return self.output