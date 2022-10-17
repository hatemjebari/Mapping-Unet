from models.forecastors.utils.simple_engine import *
from models.forecastors.Conv3DUNet.unet import *
from models.forecastors.attunet.attunet import *
from models.forecastors.MobileNet3D.mobilenet import *
from models.forecastors.utils.utils import *
from models.forecastors.utils.loss import * 
from data_prep.dataset.wrf_data import *
from models.utils.configtrain import *
from data_prep.config.env import *
from models.forecastors.ResNet3D.resnet import *
from netCDF4 import Dataset
import netCDF4 as nc
import pandas as pd
import numpy as np
import warnings
import argparse
import random
import torch
import os
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
parser.add_argument("--validation_batch_size", "-vbs",type = int,default = 4)
parser.add_argument("--train_batch_size", "-tbs",type = int,default = 4)
parser.add_argument("--experiment", "-exp",type = int,default = 0)
parser.add_argument("--loss_type", "-lt",type = str,default = "ALL")
parser.add_argument("--window_size", "-ws",type = int,default = 13)
parser.add_argument("--verbose", "-v",type = bool,default = True)
parser.add_argument("--device", "-c",type = str,default = "cpu")
parser.add_argument("--epochs", "-e",type = int,default = 1000)
parser.add_argument("--model_name","-mn",type = str ,default = "UNet")
parser.add_argument("--attention","-att",type = bool,default = False)
args = parser.parse_args()



if __name__ =="__main__" :
    def init_weights(m):
        if isinstance(m,nn.Conv3d):
            torch.nn.init.normal_(m.weight)
            #m.bias.data.fill_(0.01)
        if isinstance(m,nn.ConvTranspose3d):
            torch.nn.init.normal_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m,nn.BatchNorm3d):
            torch.nn.init.normal_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    print("Loading TrainData...")
    TrainData = WRFDataset(window_size = args.window_size,
                           solar_type = "DHI",
                           data_type = "train",
                           domaine_size='reduced',
                          )
    print("Loading ValidData...")
    ValidData = WRFDataset(window_size = args.window_size,
                           solar_type = "DHI",
                           data_type = "valid",
                           domaine_size='reduced',
                          )
    input_features = 13
    output_feature = 1
    print("Define Model...")
    if args.attention:
        print("The Model Running is Attention U-Net")
    model = ResNet18(input_features)
    #UNet(in_channels = input_features,out_channels = output_feature,attention = args.attention)

    criterion = Loss(args.loss_type)
    if (args.loss_type == "PSNR" ) or (args.loss_type == "SSIM") or (args.loss_type == "MIX"):
        print("Maximization Problem")
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=2e-3,
                                      maximize = True,
                                     )
    else:
        print("Minimization Problem")
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=2e-3,
                                      betas = (0.5,0.5),
                                      weight_decay = 1e-3,
                                     )
        
    print("Loss Type", args.loss_type)
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
    model = model.apply(init_weights)
    model = model.to(args.device)
    train_config = TrainingConfig(model,
                              criterion,
                              optimizer,
                              scheduler,
                              args.device,
                              shuffle_trainloader,
                              train_batch_size,
                              shuffle_validloader,
                              valid_batch_size,
                              epoch,
                              verbose,
                              args.experiment,
                             )
    model = model.to(args.device)
    print("Start Training")
    job = Training(train_config)
    job.fit(TrainData,ValidData)