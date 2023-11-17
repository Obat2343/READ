import torch
import torch.nn as nn
import torch.nn.functional as F

# def make_labels_and_weights(cfg, info, device):
#     if cfg.PRED_LEN == 0:
#         raise ValueError("TODO")
#     else:
#         pos_len = cfg.PRED_LEN   
    
#     negative_len = cfg.SAMPLING.NUM_NEGATIVE
#     batch_size, _ = info['mask'].shape
#     labels = torch.cat([torch.ones(batch_size, pos_len), torch.zeros(batch_size, negative_len)], 1).to(device)
#     weights = torch.cat([info['mask'], torch.ones(batch_size, negative_len)], 1).to(device)
#     return {"all":labels}, {"all":weights}

class EBM_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda', mode="quat"):
        super(EBM_Loss, self).__init__()
        self.device = device
        self.ce_loss = CELoss()
    
    def forward(self, pred_dict, mode="train"):
        loss_dict = {}
        
        ce_loss = self.ce_loss(pred_dict)
        loss_dict[f"{mode}/CE_Loss"] = ce_loss.item()
        
        loss = ce_loss
        loss_dict[f"{mode}/loss"] = loss.item()
        
        return loss, loss_dict

class Eval_score(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, query_pred_dict, gt_pred_dict, mode="val"):
        loss_dict = {}
        gt_score = gt_pred_dict["score"][:,0].to(self.device)
        query_score = query_pred_dict["score"].to(self.device)
        
        with torch.no_grad():
            query_mean_score = torch.mean(query_score, 1)
            query_min_score = query_score[:,0]
        
        diff_mean = torch.mean(gt_score - query_mean_score)
        loss_dict[f"{mode}/pos-mean"] = diff_mean.item()
        
        diff_min = torch.mean(gt_score - query_min_score)
        loss_dict[f"{mode}/pos-min"] = diff_min.item()
        return loss_dict

class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.BCELoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, output_dict, label_dict, weight_dict):
        loss_dict = {}
        for i, key in enumerate(output_dict.keys()):
            output = output_dict[key]
            output = -1 * output 
            output = self.sigmoid(output)
            label = label_dict[key]
            weight = weight_dict[key]
            loss = torch.mean(self.criteria(output, label) * weight)
            
            if i == 0:
                total_loss = loss
            else:
                total_loss += loss

            loss_dict[key] = loss.detach().cpu().item()
        return total_loss, loss_dict

class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.CrossEntropyLoss()

    def forward(self, output_dict):

        output = output_dict["score"]
        output = -1 * output

        B, N = output.shape
        device = output.device
        gt = torch.zeros(B, dtype=torch.long).to(device)

        loss = self.criteria(output, gt)

        return loss

class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2.5, reduction='mean'):
        super().__init__(self)
        self.weight=weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )