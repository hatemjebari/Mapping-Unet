from tqdm import tqdm
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torchvision
import neptune.new as neptune
from neptune.new.types import File
import torch.nn.init as weight_init
from torch.utils.data import RandomSampler
import matplotlib.pyplot as plt
from models.forecastors.utils.loss import * 
from data_prep.dataset.wrf_data import *
plt.style.use('classic')





MODELS_WEIGHTS = os.environ.get("MODELS_WEIGHTS","/home/resifis/Desktop/kaustcode/Packages/experiments/")

NEPTUNE_API_TOKENS = os.environ.get("NEPTUNE_API_TOKENS","eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDkxNzQyNS1iZjk1LTRiYzQtYmY3OS1jMzBkZjA4ZDhkNTAifQ==")


ssim = Loss("SSIM")
psnr = Loss("PSNR")

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Training():
    def __init__(self,train_config):
        self.train_config = train_config
        
    def _initialize_weights(self,net):
        for name, param in net.named_parameters(): 
            weight_init.normal_(param)
        
    def train_fn(self,model,train_loader):
        self.train_config.model.train()
        l2_lambda = 1e-5
        tr_loss = 0
        counter = 0
        losses = AverageMeter()
        tqt = tqdm(enumerate(train_loader),total = len(train_loader))
        for index,train_batch in tqt:
            data = train_batch["data"].to(self.train_config.device)
            target = train_batch["target"].to(self.train_config.device)
            self.train_config.optimizer.zero_grad()
            pred_target = model(data)
            pred_target = pred_target.squeeze(1)
            train_loss = self.train_config.criterion(pred_target,target)
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #train_loss = train_loss + l2_lambda*l2_norm
            train_loss.backward()
            self.train_config.optimizer.step()
            tr_loss += train_loss.item()        
            counter = counter + 1
            ss = ssim(pred_target,target).item()
            ps = psnr(pred_target,target).item()
            losses.update(train_loss.item(),pred_target.size(0))
            tqt.set_postfix(Loss = losses.avg, Batch_number = index,SSIM = ss,PSNR = ps )
        return losses.avg

    def valid_fn(self,model,validation_loader):
        self.train_config.model.eval()
        val_loss = 0
        counter = 0
        l2_lambda = 1e-5
        losses = AverageMeter()
        tqt = tqdm(enumerate(validation_loader),total = len(validation_loader))
        with torch.no_grad():
            for index, valid_batch in tqt :
                data = valid_batch["data"].to(self.train_config.device)
                target = valid_batch["target"].to(self.train_config.device)
                self.train_config.optimizer.zero_grad()
                pred_target = model(data)
                pred_target = pred_target.squeeze(1)
                validation_loss = self.train_config.criterion(pred_target,target)
                #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                #validation_loss = validation_loss + l2_lambda*l2_norm
                val_loss += validation_loss.item()        
                counter = counter + 1
                ss = ssim(pred_target,target).item()
                ps = psnr(pred_target,target).item()
                losses.update(validation_loss.item(),pred_target.size(0))
                tqt.set_postfix(loss = losses.avg, batch_number = index,SSIM = ss,PSNR = ps)
        return losses.avg
    
    
    def get_dataloader(self,train_dataset,valid_dataset):
        
        train_sampler = RandomSampler(train_dataset,replacement = False,num_samples = 2000)
        valid_sampler = RandomSampler(valid_dataset,replacement = False,num_samples = 200)
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        shuffle = self.train_config.shuffle_trainloader,
                                                        batch_size = self.train_config.train_batch_size,
                                                        sampler = train_sampler,
                                                       )
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        shuffle = self.train_config.shuffle_validloader,
                                                        batch_size = self.train_config.valid_batch_size,
                                                        sampler = valid_sampler,
                                                       )
        return train_data_loader,valid_data_loader
    
    @staticmethod
    def checkpoints(epoch,model,optimizer,loss,exp):
        path = os.path.join(MODELS_WEIGHTS,f"weights-{exp}/epoch_{epoch}.pt")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)
        
        
    @staticmethod
    def test (model,device,run):
        item = 985
        TrainData = WRFDataset(window_size = 250,solar_type = "DHI", data_type = "train",domaine_size='reduced')
        sample = TrainData[item]['data'].unsqueeze(0).to(device)
        truth = TrainData[item]['target']
        pred = model(sample)
        pred = pred.squeeze(0).squeeze(0)
        pred = pred.detach().cpu()
        exp_id = 20
        exp_true = truth[exp_id]
        exp_pred = pred[exp_id]
        
        figure1,ax1= plt.subplots(figsize=(20, 10))
        im1 = ax1.imshow(exp_true)
        cax = plt.axes([0.85, 0.1, 0.05, 0.8])
        plt.colorbar(im1,cax=cax)
        #plt.clf()
        
        figure2,ax2= plt.subplots(figsize=(20, 10))
        im2 = ax2.imshow(exp_pred)
        cax = plt.axes([0.85, 0.1, 0.05, 0.8])
        plt.colorbar(im2,cax=cax)
        #plt.clf()
        
        lat = 40
        log = 120
        sdata = sample.squeeze(0)
        cloud = sdata[0]
        cloud = cloud[:,lat,log].detach().cpu()
        seq_target = truth[:,lat,log]
        seq_pred = pred[:,lat,log]
        seq_pred[seq_pred<0] = 0
        
        
        figure3,ax3 = plt.subplots(figsize = (20,10))
        ax3.plot(seq_target, color="Green", label = "Ground Truth GHI(W/M2)")
        ax3.set_xlabel("Time",fontsize=14)
        ax3.set_title("Ground Reference vs Predicted GHI (W/M2)",fontsize = 25)
        ax3.set_ylabel("GHI (W/m2)",color="red",fontsize=25)
        ax3.plot(seq_pred, color="red", label = "Predicted GHI (W/M2)")
        ax3.set_xlabel("Time",fontsize=14)
        ax3.set_ylabel("GHI (W/m2)",color="red",fontsize=25)
        plt.grid(True)
        plt.legend()
        ax4=ax3.twinx()
        ax4.plot(100*cloud,color="blue",label = f"Cloud%")
        ax4.set_ylabel("Clouds cover (%)",color="blue",fontsize=25)
        plt.grid(True)
        
        print("...Upload to Neptune.ai...")
        run[f"Train - Ground Reference (WRF)"].upload(neptune.types.File.as_image(figure1))
        run[f"Train - Predicted"].upload(neptune.types.File.as_image(figure2))
        run[f"Train - Predicted-OverTime"].upload(neptune.types.File.as_image(figure3))

        
    def fit(self,train_dataset,valid_dataset):
        train_loss = []
        valid_loss = []
        best = 5000
        run = neptune.init(project="sofienresifi1997/KaustProject",
                           api_token=NEPTUNE_API_TOKENS,
                          )

        
        for epoch in range(self.train_config.epoch):
            train_data_loader,valid_data_loader = self.get_dataloader(train_dataset,valid_dataset)
            Training.test(self.train_config.model,self.train_config.device,run)
            if self.train_config.verbose :
                print(f".........EPOCH {epoch}........")
            tr_loss = self.train_fn(self.train_config.model,train_data_loader)
            train_loss.append(tr_loss)
            run["Train/loss"].log(tr_loss)
            if self.train_config.verbose :
                print(f".........Train Loss = {tr_loss}........")
            val_loss = self.valid_fn(self.train_config.model,valid_data_loader)
            valid_loss.append(val_loss)
            run["valid/loss"].log(val_loss)
            Training.checkpoints(epoch,self.train_config.model,self.train_config.optimizer,val_loss,self.train_config.exp)
           
            self.train_config.scheduler.step(val_loss)
            if self.train_config.verbose:
                print(f"...........Validation Loss = {val_loss}.......")

            if val_loss < best :
                best = val_loss
                patience = 0
            else:
                print("Score is not improving with patient = ",patience)
                patience +=1

            if patience >= self.train_config.epoch:
                print(f"Early Stopping on Epoch {epoch}")
                print(f"Best Loss = {best}")
                break
                
            
                
        
        PATH = os.path.join(MODELS_WEIGHTS,"model_300.pth")
        torch.save(self.train_config.model.state_dict(),PATH)
        self.train_config.model.load_state_dict(torch.load(PATH))