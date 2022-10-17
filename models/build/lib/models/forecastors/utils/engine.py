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
plt.style.use('classic')





MODELS_WEIGHTS = os.environ.get("MODELS_WEIGHTS","/home/resifis/Desktop/kaustcode/Packages/weights")
NEPTUNE_API_TOKENS = os.environ.get("NEPTUNE_API_TOKENS",               "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDkxNzQyNS1iZjk1LTRiYzQtYmY3OS1jMzBkZjA4ZDhkNTAifQ==")



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
        
    def train_fn(self,model,train_loader,run,sparse = dict()):
        self.train_config.model.train()
        
        tr_loss = 0
        counter = 0
        losses = AverageMeter()
        tqt = tqdm(enumerate(train_loader),total = len(train_loader))
        for index,train_batch in tqt:
            list_sparse = []
            data = train_batch["data"].to(self.train_config.device)
            target = train_batch["target"].to(self.train_config.device)
            self.train_config.optimizer.zero_grad()
            pred_target = model(data)
            pred_target = pred_target.squeeze(1)
            train_loss = self.train_config.criterion(pred_target,target)
            train_loss.backward()
            self.train_config.optimizer.step()
            tr_loss += train_loss.item()        
            counter = counter + 1
            losses.update(train_loss.item(),pred_target.size(0))
            run["train/loss"].log(losses.avg)
            if index%3 == 0 :
                ind = 2
                im_true = target[0,:,:,:][ind].detach().cpu()
                im_target = pred_target[0,:,:,:][ind].detach().cpu()
                im_hcloud = data[0,:,:,:][0][ind].detach().cpu()
                im_aod = data[0,:,:,:][3][ind].detach().cpu()
                im_ozone = data[0,:,:,:][4][ind].detach().cpu()
                im_water_vapor = data[0,:,:,:][5][ind].detach().cpu()

                figure1, ax1 = plt.subplots(figsize=(20, 10))
                im1 = ax1.imshow(im_true)
                cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                plt.colorbar(im1,cax=cax)
                plt.clf()

                figure2, ax2 = plt.subplots(figsize=(20, 10))
                im2 = ax2.imshow(im_target)
                cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                plt.colorbar(im2,cax=cax)
                plt.clf()

                figure3, ax3 = plt.subplots(figsize=(20, 10))
                im3 = ax3.imshow(im_hcloud)
                cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                plt.colorbar(im3,cax=cax)
                plt.clf()

                figure4, ax4 = plt.subplots(figsize=(20, 10))
                im4 = ax4.imshow(im_aod)
                cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                plt.colorbar(im4,cax=cax)
                plt.clf()

                figure5, ax5 = plt.subplots(figsize=(20, 10))
                im5 = ax5.imshow(im_ozone)
                cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                plt.colorbar(im5,cax=cax)
                plt.clf()

                figure6, ax6 = plt.subplots(figsize=(20, 10))
                im6 = ax6.imshow(im_water_vapor)
                cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                plt.colorbar(im6,cax=cax)
                plt.clf()

                run[f"Train - Ground Truth"].upload(neptune.types.File.as_image(figure1))
                run[f"Train - Predicted"].upload(neptune.types.File.as_image(figure2))
                run[f"Train - High Clouds"].upload(neptune.types.File.as_image(figure3))
                run[f"Train - Aerosols(AOD)"].upload(neptune.types.File.as_image(figure4))
                run[f"Train - Ozone"].upload(neptune.types.File.as_image(figure5))
                run[f"Train - Water Vapor"].upload(neptune.types.File.as_image(figure6))
                plt.close()
            tqt.set_postfix(Loss = losses.avg, Batch_number = index )
        return tr_loss/counter

    def valid_fn(self,model,validation_loader,run,sparse = dict()):
        self.train_config.model.eval()
        val_loss = 0
        counter = 0
        losses = AverageMeter()
        tqt = tqdm(enumerate(validation_loader),total = len(validation_loader))
        with torch.no_grad():
            for index, valid_batch in tqt :
                list_sparse = []
                data = valid_batch["data"].to(self.train_config.device)
                target = valid_batch["target"].to(self.train_config.device)
                self.train_config.optimizer.zero_grad()
                pred_target = model(data)
                pred_target = pred_target.squeeze(1)

                accuracy = pixelAcc(pred_target,target)
                validation_loss = self.train_config.criterion(pred_target,target)
                val_loss += validation_loss.item()        
                counter = counter + 1
                losses.update(validation_loss.item(),pred_target.size(0))
                rrmse = RRMSE(pred_target,target)
                run["valid/loss"].log(losses.avg)
                if index%20 == 0 :
                    ind = 2
                    im_true = target[0,:,:,:][ind].detach().cpu()
                    im_target = pred_target[0,:,:,:][ind].detach().cpu()
                    im_hcloud = data[0,:,:,:][0][ind].detach().cpu()
                    im_aod = data[0,:,:,:][3][ind].detach().cpu()
                    im_ozone = data[0,:,:,:][4][ind].detach().cpu()
                    im_water_vapor = data[0,:,:,:][5][ind].detach().cpu()

                    figure1, ax1 = plt.subplots(figsize=(20, 10))
                    im1 = ax1.imshow(im_true)
                    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                    plt.colorbar(im1,cax=cax)
                    plt.clf()

                    figure2, ax2 = plt.subplots(figsize=(20, 10))
                    im2 = ax2.imshow(im_target)
                    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                    plt.colorbar(im2,cax=cax)
                    plt.clf()


                    figure3, ax3 = plt.subplots(figsize=(20, 10))
                    im3 = ax3.imshow(im_hcloud)
                    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                    plt.colorbar(im3,cax=cax)
                    plt.clf()

                    figure4, ax4 = plt.subplots(figsize=(20, 10))
                    im4 = ax4.imshow(im_aod)
                    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                    plt.colorbar(im4,cax=cax)
                    plt.clf()

                    figure5, ax5 = plt.subplots(figsize=(20, 10))
                    im5 = ax5.imshow(im_ozone)
                    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                    plt.colorbar(im5,cax=cax)
                    plt.clf()

                    figure6, ax6 = plt.subplots(figsize=(20, 10))
                    im6 = ax6.imshow(im_water_vapor)
                    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
                    plt.colorbar(im6,cax=cax)
                    plt.clf()

                    run[f"Validation - Ground Truth"].upload(neptune.types.File.as_image(figure1))
                    run[f"Validation - Predicted"].upload(neptune.types.File.as_image(figure2))
                    run[f"Validation - High Clouds"].upload(neptune.types.File.as_image(figure3))
                    run[f"Validation - Aerosols(AOD)"].upload(neptune.types.File.as_image(figure4))
                    run[f"Validation - Ozone"].upload(neptune.types.File.as_image(figure5))
                    run[f"Validation - Water Vapor"].upload(neptune.types.File.as_image(figure6))
                    plt.close()
                tqt.set_postfix(loss = losses.avg, batch_number = index)
        return val_loss/counter
    
    
    def get_dataloader(self,train_dataset,valid_dataset):
        
        train_sampler = RandomSampler(train_dataset,replacement = True,num_samples = 2000)
        valid_sampler = RandomSampler(valid_dataset,replacement = True,num_samples = 1000)
        
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
    def checkpoints(epoch,model,optimizer,loss):
        path = os.path.join(MODELS_WEIGHTS,f"epoch_{epoch}.pt")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)
        
    def fit(self,train_dataset,valid_dataset):
        train_loss = []
        valid_loss = []
        sparse = dict()
        sparse['lat'] = train_dataset.lat
        sparse['lon'] = train_dataset.lon
        best = 5000
        run = neptune.init(
            project="sofienresifi1997/KaustProject",
            api_token=NEPTUNE_API_TOKENS,
        )  
        for epoch in range(self.train_config.epoch):
            train_data_loader,valid_data_loader = self.get_dataloader(train_dataset,valid_dataset)
            if self.train_config.verbose :
                print(f".........EPOCH {epoch}........")
            tr_loss = self.train_fn(self.train_config.model,train_data_loader,run,sparse)
            train_loss.append(tr_loss)
            if self.train_config.verbose :
                print(f".........Train Loss = {tr_loss}........")
            val_loss = self.valid_fn(self.train_config.model,valid_data_loader,run,sparse)
            valid_loss.append(val_loss)
            Training.checkpoints(epoch,self.train_config.model,self.train_config.optimizer,val_loss)
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
                
        
        PATH = os.path.join(MODELS_WEIGHTS,"model1.pth")
        torch.save(self.train_config.model.state_dict(),PATH)
        model.load_state_dict(torch.load(PATH))