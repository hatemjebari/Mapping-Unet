import os
MODELS_WEIGHTS = os.environ.get("MODELS_WEIGHTS","/home/resifis/Desktop/kaustcode/Packages/models/src/models/weights/")


class TrainingConfig():
    def __init__(self,
                 model,
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
                 exp,
                 ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.shuffle_trainloader = shuffle_trainloader
        self.train_batch_size = train_batch_size
        self.shuffle_validloader = shuffle_validloader
        self.valid_batch_size = valid_batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.exp = exp
        
        
    def _initialize_weights(self,m):
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
        elif isinstance(m, nn.MaxPool3d):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)