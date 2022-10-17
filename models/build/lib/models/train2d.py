from models.forecastors.GeneralUnet.general_unet import *
from models.forecastors.utils.loss import * 
from data_prep.dataset.dataset import *
from models.utils.configtrain import *
from data_prep.config.env import *
import matplotlib.pyplot as plt
import torch.optim as optim
from netCDF4 import Dataset
from skimage import filters
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import argparse
import random
import torch
import os
from models.forecastors.utils.general_engine import *
warnings.filterwarnings("ignore")


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True 
setup_seed(15)




parser = argparse.ArgumentParser()
parser.add_argument("--shuffle_trainloader", "-stl",type = bool,default = False)
parser.add_argument("--shuffle_validloader", "-svl",type = bool,default = False)
parser.add_argument("--validation_batch_size", "-vbs",type = int,default = 6)
parser.add_argument("--train_batch_size", "-tbs",type = int,default = 6)
parser.add_argument("--loss_type", "-lt",type = str,default = "RMSE")
parser.add_argument("--verbose", "-v",type = bool,default = True)
parser.add_argument("--device", "-c",type = str,default = "cpu")
parser.add_argument("--epochs", "-e",type = int,default = 1000)
args = parser.parse_args()




if __name__ =="__main__" :
    print("Load Datasets")
    DataSet = Dataset2D("GHI")
    def init_weights(m):
        if isinstance(m,nn.Conv2d):
            torch.nn.init.normal(m.weight)
        if isinstance(m,nn.ConvTranspose2d):
            torch.nn.init.normal(m.weight)
        if isinstance(m,nn.BatchNorm2d):
            torch.nn.init.normal(m.weight)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    input_features = 14
    output_feature = 1
    print("Load The Model")
    model = UNet2D(in_channels = input_features,
               out_channels = output_feature,
            )
    
    criterion = Loss("L1Loss")
    optimizer = optim.AdamW(model.parameters(), lr=0.001,betas = (0.5,0.5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor = 0.2,
                                                           patience = 3,
                                                           verbose = True)

    shuffle_trainloader = args.shuffle_trainloader
    train_batch_size = args.train_batch_size
    shuffle_validloader = args.shuffle_validloader
    valid_batch_size = args.validation_batch_size
    epoch = args.epochs
    verbose = args.verbose
    device = args.device
    train_config = TrainingConfig(model,
                              criterion,
                              optimizer,
                              scheduler,
                              device,
                              shuffle_trainloader,
                              train_batch_size,
                              shuffle_validloader,
                              valid_batch_size,
                              epoch,
                              verbose,
                             )
    model = model.to(device)
    model = model.apply(init_weights)
    print('Start Training')
    job = Training(train_config)
    job.fit(DataSet,DataSet)