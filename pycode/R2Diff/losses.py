import torch

class Diffusion_Loss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
    
    def forward(self, pred_dict, noise_dict, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_dict.keys():
            loss = self.l1(pred_dict[key], noise_dict[key])
            loss_dict[f"{mode}/{key}"] = loss.item()
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict

class DSM_Loss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss(reduction="none")
    
    def forward(self, pred_dict, scaled_noise_dict, stds, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_dict.keys():
            loss = self.l1(pred_dict[key], -scaled_noise_dict[key])
            loss = torch.mean(loss.view(loss.shape[0], -1), 1)
            loss_dict[f"{mode}/{key}"] = torch.mean(loss).item()
            loss = torch.mean(loss * torch.reciprocal(stds))
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict
    
class DSM_Loss_DiffusionLike(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
    
    def forward(self, pred_dict, noise_dict, stds, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_dict.keys():
            loss = self.l1(pred_dict[key], -noise_dict[key])
            loss_dict[f"{mode}/{key}"] = loss.item()
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict