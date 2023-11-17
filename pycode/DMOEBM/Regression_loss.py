import torch
import pytorch3d
import numpy as np
from einops import repeat, rearrange
from scipy.spatial.transform import Rotation as R

class Rotation_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda', mode="quat"):
        super(Rotation_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.device = device
        self.mode = mode
        if mode not in ["quat", "6d"]:
            raise ValueError("TODO")
        
    def forward(self, pred_dict, gt_dict):
        pred_rot = pred_dict["rotation"].to(self.device)
        B, N, D = pred_rot.shape
        if self.mode == "quat":
            pred_rot = pytorch3d.transforms.standardize_quaternion(pred_rot)
            gt_rot = gt_dict["rotation"].to(self.device)
        elif self.mode == "6d":
            pred_rot = pytorch3d.transforms.rotation_6d_to_matrix(pred_rot)
            gt_rot = pytorch3d.transforms.rotation_6d_to_matrix(gt_dict["rotation"].to(self.device))
        
        loss = self.MSE(pred_rot, gt_rot)
        return loss 

class UV_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(UV_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_uv = pred_dict["uv"].to(self.device)
        B, N, D = pred_uv.shape
        gt_uv = gt_dict["uv"].to(self.device)
        loss = self.MSE(pred_uv, gt_uv)
        return loss

class Z_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Z_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_z = pred_dict["z"].to(self.device)
        B, N, D = pred_z.shape
        gt_z = gt_dict["z"].to(self.device)
        loss = self.MSE(pred_z, gt_z)
        return loss

class Grasp_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Grasp_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_grasp = pred_dict["grasp_state"].to(self.device)
        B, N, D = pred_grasp.shape
        gt_grasp = gt_dict["grasp_state"].to(self.device)
        
        loss = self.MSE(pred_grasp, gt_grasp)
        return loss 

class Motion_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda', mode="quat"):
        super(Motion_Loss, self).__init__()
        self.device = device
        self.rot_loss = Rotation_Loss(device, mode=mode)
        self.uv_loss = UV_Loss(device)
        self.z_loss = Z_Loss(device)
        self.grasp_loss = Grasp_Loss(device)
    
    def forward(self, pred_dict, gt_dict, mode="train"):
        loss_dict = {}
        
        rot_loss = self.rot_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/rot_loss"] = rot_loss.item()
        
        uv_loss = self.uv_loss(pred_dict, gt_dict) * 10
        loss_dict[f"{mode}/uv_loss"] = uv_loss.item()
        
        z_loss = self.z_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/z_loss"] = z_loss.item()
        
        grasp_loss = self.grasp_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/grasp_loss"] = grasp_loss.item()
        
        loss = rot_loss + uv_loss + z_loss + grasp_loss
        loss_dict[f"{mode}/loss"] = loss.item()
        
        return loss, loss_dict

class Iterative_Motion_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda', mode="quat"):
        super(Iterative_Motion_Loss, self).__init__()
        self.device = device
        self.rot_loss = Rotation_Loss(device, mode=mode)
        self.uv_loss = UV_Loss(device)
        self.z_loss = Z_Loss(device)
        self.grasp_loss = Grasp_Loss(device)
    
    def forward(self, pred_dict_list, gt_dict, mode="train"):
        loss_dict = {}
        
        rot_loss_sum = 0
        uv_loss_sum = 0
        z_loss_sum = 0
        grasp_loss_sum = 0
        for i, pred_dict in enumerate(pred_dict_list):
            rot_loss = self.rot_loss(pred_dict, gt_dict)
            loss_dict[f"{mode}/iterate_{i}_rot_loss"] = rot_loss.item()
            rot_loss_sum += rot_loss
            
            uv_loss = self.uv_loss(pred_dict, gt_dict) * 10
            loss_dict[f"{mode}/iterate_{i}_uv_loss"] = uv_loss.item()
            uv_loss_sum += uv_loss

            z_loss = self.z_loss(pred_dict, gt_dict)
            loss_dict[f"{mode}/iterate_{i}_z_loss"] = z_loss.item()
            z_loss_sum += z_loss

            grasp_loss = self.grasp_loss(pred_dict, gt_dict)
            loss_dict[f"{mode}/iterate_{i}_grasp_loss"] = grasp_loss.item()
            grasp_loss_sum += grasp_loss

        loss_dict[f"{mode}/rot_loss"] = rot_loss_sum.item() / len(pred_dict_list)
        loss_dict[f"{mode}/uv_loss"] = uv_loss_sum.item() / len(pred_dict_list)
        loss_dict[f"{mode}/z_loss"] = z_loss_sum.item() / len(pred_dict_list)
        loss_dict[f"{mode}/grasp_loss"] = grasp_loss_sum.item() / len(pred_dict_list)
        
        loss = (rot_loss_sum + uv_loss_sum + z_loss_sum + grasp_loss_sum) / len(pred_dict_list)
        loss_dict[f"{mode}/loss"] = loss.item()
        
        return loss, loss_dict
### Evaluation

class RMSE_Pose_Eval(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(RMSE_Pose_Eval, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.device = device
    
    def forward(self, pred_dict, gt_dict):
        pred_pos = pred_dict["pos"].to(self.device)
        B, N, D = pred_pos.shape
        gt_pos = gt_dict["pos"].to(self.device)
        
        pred_pos = pred_pos * 1000
        gt_pos = gt_pos * 1000

        loss = self.MSE(pred_pos, gt_pos)
        loss = torch.sqrt(torch.sum(loss.view(B,N,-1), 2))
        loss = torch.mean(loss)
        return loss

class RMSE_Z_Eval(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(RMSE_Z_Eval, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.device = device
    
    def forward(self, pred_dict, gt_dict):
        pred_pos = pred_dict["z"].to(self.device)
        B, N, D = pred_pos.shape
        gt_pos = gt_dict["z"].to(self.device)
        
        pred_pos = pred_pos * 1000
        gt_pos = gt_pos * 1000

        loss = self.MSE(pred_pos, gt_pos)
        loss = torch.sqrt(torch.sum(loss.view(B,N,-1), 2))
        loss = torch.mean(loss)
        return loss

class Rotation_Eval(torch.nn.Module):
    
    def __init__(self, rot_mode="quat"):
        super(Rotation_Eval, self).__init__()
        self.rot_mode = rot_mode
        
    def forward(self, pred_dict, gt_dict):
        pred_rot = pred_dict["rotation"]
        gt_rot = gt_dict["rotation"]

        pred_rot = self.convert_rotation(pred_rot)
        gt_rot = self.convert_rotation(gt_rot)
        
        inv_rot = pred_rot.inv()
        diff_rot_r = gt_rot * inv_rot

        diff_rot = diff_rot_r.magnitude()
        diff_angle = np.rad2deg(diff_rot)
        error = np.mean(diff_angle)
        return error
    
    def convert_rotation(self, rot):
        rot = rot.detach().cpu()
        rot_np = rearrange(rot, "B N ... -> (B N) ...").numpy()

        if self.rot_mode == "quat":
            rot_r = R.from_quat(rot_np)
        elif self.rot_mode in ["6d","matrix"]:
            rot_r = R.from_matrix(rot_np)
        else:
            raise ValueError("TODO")

        return rot_r

# class RMSE_Rotation_Eval(torch.nn.Module):
    
#     def __init__(self, device='cuda'):
#         super(RMSE_Rotation_Eval, self).__init__()
#         self.MSE = torch.nn.MSELoss(reduction='none')
#         self.device = device
        
#     def forward(self, pred_dict, gt_dict):
#         pred_rot = pred_dict["rotation"].to(self.device)
#         B, N, D = pred_rot.shape
#         pred_rot = rearrange(pred_rot, 'B N D -> (B N) D')
#         pred_rot = compute_rm(pred_rot)
#         pred_rot = rearrange(pred_rot, '(B D) N M -> B D N M', B=B)
#         gt_rot = gt_dict["rotation"].to(self.device)
#         mask = gt_dict["mask"].to(self.device)
        
#         loss = self.MSE(pred_rot, gt_rot)
#         loss = torch.sqrt(torch.sum(loss.view(B,N,-1), 2)) * mask
#         loss = torch.sum(loss, 1)
#         length = torch.sum(mask, 1)
#         loss = torch.mean(loss / length)
#         return loss 

class Accuracy_Grasp_Eval(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Accuracy_Grasp_Eval, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_grasp = pred_dict["grasp_state"].to(self.device)
        B, N, D = pred_grasp.shape
        pred_grasp = (pred_grasp>0.5).float()
        gt_grasp = gt_dict["grasp_state"].to(self.device)
        gt_grasp = (gt_grasp>0.5).float()
        loss = self.L1(pred_grasp, gt_grasp)
        return 1 - loss

class Evaluation(torch.nn.Module):
    
    def __init__(self, device='cuda', mode="quat"):
        super(Evaluation, self).__init__()
        self.device = device
        self.rot_loss = Rotation_Eval(mode)
        self.pos_loss = RMSE_Pose_Eval(device)
        self.z_loss = RMSE_Z_Eval(device)
        self.grasp_loss = Accuracy_Grasp_Eval(device)
    
    def forward(self, pred_dict, gt_dict, mode="train"):
        loss_dict = {}
        
        rot_loss = self.rot_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/rot_error"] = rot_loss
        
        pos_loss = self.pos_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/pos_error"] = pos_loss.item()
        
        z_loss = self.z_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/z_error"] = z_loss.item()
        
        grasp_loss = self.grasp_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/grasp_accuracy"] = grasp_loss.item()
        
        return loss_dict
    
def get_pos(query, intrinsic, image_size=(256,256)):
    uv = query["uv"]
    z = query["z"]
    h, w = image_size
    u, v = uv[:,:,0], uv[:,:,1]
    u = (u + 1) / 2 * (w - 1)
    v = (v + 1) / 2 * (h - 1)
    uv_denorm = torch.stack([u, v], 2)
    ones = torch.ones(*uv_denorm.shape[:-1], 1)
    uv_denorm = torch.cat([uv_denorm, ones], 2)
    intrinsic = torch.tensor(intrinsic, dtype=torch.float)
    inv_intrinsic = torch.linalg.inv(intrinsic)

    xy = torch.einsum('ij,bkj->bki', inv_intrinsic, uv_denorm)
    xyz = xy * z
    query["pos"] = xyz
    return query