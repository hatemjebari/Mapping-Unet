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





MODELS_WEIGHTS = os.environ.get("MODELS_WEIGHTS","/home/resifis/Desktop/kaustcode/Packages/weights2d")


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
        self.model = self.train_config.model
        self.device = self.train_config.device
        self.optimizer = self.train_config.optimizer
        self.tbs = self.train_config.train_batch_size
        self.vbs = self.train_config.valid_batch_size
        self.shuffle_trainloader = self.train_config.shuffle_trainloader
        self.shuffle_validloader = self.train_config.shuffle_validloader
        self.verbose = self.train_config.verbose
        self.epoch = self.train_config.epoch
        self.criterion = self.train_config.criterion
        self.scheduler = self.train_config.scheduler
        
        
    def _initialize_weights(self,net):
        for name, param in net.named_parameters(): 
            weight_init.normal_(param)
        
    def train_fn(self,model,train_loader):
        model.train()
        
        tr_loss = 0
        counter = 0
        losses = AverageMeter()
        tqt = tqdm(enumerate(train_loader),total = len(train_loader))
        for index,train_batch in tqt:
            list_sparse = []
            data = train_batch["data"].to(self.device)
            target = train_batch["target"].to(self.device)
            self.optimizer.zero_grad()
            pred_target = model(data)
            pred_target = pred_target.squeeze(1)
            train_loss = self.criterion(pred_target,target)
            train_loss.backward()
            self.optimizer.step()
            tr_loss += train_loss.item()        
            counter = counter + 1
            losses.update(train_loss.item(),pred_target.size(0))
            tqt.set_postfix(Loss = losses.avg, Batch_number = index )
        return tr_loss/counter

    def valid_fn(self,model,validation_loader):
        model.eval()
        val_loss = 0
        counter = 0
        losses = AverageMeter()
        tqt = tqdm(enumerate(validation_loader),total = len(validation_loader))
        with torch.no_grad():
            for index, valid_batch in tqt :
                list_sparse = []
                data = valid_batch["data"].to(self.device)
                target = valid_batch["target"].to(self.device)
                self.optimizer.zero_grad()
                pred_target = model(data)
                pred_target = pred_target.squeeze(1)
                validation_loss = self.criterion(pred_target,target)
                val_loss += validation_loss.item()        
                counter = counter + 1
                losses.update(validation_loss.item(),pred_target.size(0))
                tqt.set_postfix(loss = losses.avg, batch_number = index)
        return losses.avg
    
    
    
    @staticmethod
    def checkpoints(epoch,model,optimizer,loss):
        path = os.path.join(MODELS_WEIGHTS,f"epoch_{epoch}.pt")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)
        
    def get_dataloader(self,train_dataset,valid_dataset):

        train_sampler = RandomSampler(train_dataset,replacement = True,num_samples = 1500)
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
        
    def fit(self,train_dataset,valid_dataset):
        train_loss = []
        valid_loss = []
        best = 5000

            
        for epoch in range(self.epoch):
            train_data_loader,valid_data_loader = self.get_dataloader(train_dataset,valid_dataset)
            if self.verbose :
                print(f".........EPOCH {epoch}........")
            tr_loss = self.train_fn(self.model,train_data_loader)
            train_loss.append(tr_loss)
            if self.verbose :
                print(f".........Train Loss = {tr_loss}........")
            val_loss = self.valid_fn(self.model,valid_data_loader)
            valid_loss.append(val_loss)
            Training.checkpoints(epoch,self.model,self.optimizer,val_loss)
            self.scheduler.step(val_loss)
            if self.verbose:
                print(f"...........Validation Loss = {val_loss}.......")

            if val_loss < best :
                best = val_loss
                patience = 0
            else:
                print("Score is not improving with patient = ",patience)
                patience +=1

            if patience >= self.epoch:
                print(f"Early Stopping on Epoch {epoch}")
                print(f"Best Loss = {best}")
                break
                
        
        PATH = os.path.join(MODELS_WEIGHTS,"model_2d.pth")
        torch.save(self.model.state_dict(),PATH)
        self.model.load_state_dict(torch.load(PATH))