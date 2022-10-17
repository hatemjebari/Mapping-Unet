import torch


class GetParameters():
    def __init__(self,params_encoder,params_decoder,device):
        self.params_encoder =params_encoder
        self.params_decoder = params_decoder
        self.device = device
        
    def _get_params(self):
        params_model = dict()
        params_model["device"] = torch.device(self.device)
        params_model["params_encoder"] = self.params_encoder
        params_model["params_decoder"] =  self.params_decoder
        return params_model