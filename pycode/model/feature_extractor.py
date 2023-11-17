"""
Class for extract feature from feature map

input
x: torch.tensor -> shape:B,C,H,W (feature map)
uv: torch.tensor -> shape:B,N,2 (N is number of sample including negative sample)
"""
import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from einops import rearrange, reduce, repeat

from .base_module import SoftArgmax2D, Flatten

class Image_feature_extractor_model(torch.nn.Module):
    def __init__(self, extractor_name, img_feature_dim, down_scale, num_vec=8):
        super().__init__()
        
        if "query_uv_feature" == extractor_name:
            self.extractor = query_uv_extractor(down_scale)
        elif "query_uv_feature_with_PE" == extractor_name:
            self.extractor = query_uv_extractor(down_scale, pos_emb=True, dim=img_feature_dim)
        elif "ViT_feature" == extractor_name:
            self.extractor = ViT_extraxtor(img_feature_dim)
        elif "gap" in extractor_name:
            self.extractor = gap()
        elif "softargmax_uv" in extractor_name:
            self.extractor = softargmax_uv(img_feature_dim, down_scale, num_vec)
            raise ValueError("TODO: now coding")
        elif "softargmax_feature" in extractor_name:
            self.extractor = softargmax_feature(img_feature_dim, down_scale, num_vec)
            raise ValueError("TODO: now coding")
        else:
            raise ValueError(f"Invalid key: {extractor_name} is invalid key for the Image-feature_extractor_model (in feature_extractory.py)")
        
    def forward(self, x, uv):
        extractor_dict, extractor_info = self.extractor(x, uv)
        return extractor_dict, extractor_info

class PositionalEncoding(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 65536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class query_uv_extractor(nn.Module):
    
    def __init__(self, down_scale, do_norm=False, pos_emb=False, dim=0):
        super().__init__()
        self.down_scale = down_scale
        self.norm = do_norm

        if pos_emb:
            self.pos_emb = PositionalEncoding(dim)
            if dim == 0:
                raise ValueError("Invalid dim")
            self.dim = dim
        else:
            self.dim = 0
            
    def forward(self,x,y):
        """
        Input 
        x: feature B,C,H,W
        y: pose B,N,S,2

        Output
        output_dict: dict
            key:
                img_feature: feature of image, shape(B, N, C)

        Note
        B: batch size
        C: Num channel
        H: Height
        W: Width
        N: Num query
        """
        debug_info = {}
        output_dict = {}
        B,C,H,W = x.shape

        if self.norm:
            down_y = y / self.down_scale
            y = self.pos_norm(down_y,H,W) # B, N, S, 2
            
        # if self.dim != 0:
        #     x = rearrange(x, "B C H W -> B (H W) C")
            
        #     x = rearrange(x, "B (H W) C -> B C H W",H=H)

        feature = torch.nn.functional.grid_sample(x, y, mode='bilinear', padding_mode='zeros', align_corners=True) # B, C, N, S
        feature = rearrange(feature, 'B C N S -> B N S C') # B, N, S, C
        output_dict["img_feature"] = feature
        return output_dict, debug_info
    
    def pos_norm(self,pos,H,W):
        x_coords = pos[:,:,0]
        y_coords = pos[:,:,1]
        
        x_coords = (x_coords / W) * 2 - 1
        y_coords = (y_coords / H) * 2 - 1
        
        return torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)

class gap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = Flatten()
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x,y):
        """
        Input 
        x: feature B,C,H,W
        y: pose B,N,S,2

        Output
        output_dict: dict
            key:
                img_feature: feature of image, shape(B N C)

        Note
        B: batch size
        C: Num channel
        H: Height
        W: Width
        N: Num query
        """
        output_dict = {}
        debug_info = {}
        _, N, _, _ = y.shape
        x = self.gap(x) # B, C, 1, 1
        x = self.flat(x) # B, C
        x = repeat(x, 'B C -> B N C', N=N) # B, N, C
        output_dict["img_feature"] = x
        return output_dict, debug_info

class ViT_extraxtor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down_scale = nn.Conv2d(dim, dim, 16, 16, 0)
        self.pos_emb = PositionalEncoding(dim, max_len=256)

    def forward(self,x,y):
        output_dict = {}
        debug_info = {}

        _, N, _, _ = y.shape
        x = self.down_scale(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        x = self.pos_emb(x)
        x = repeat(x, 'B S C -> B N S C', N=N)
        output_dict["img_feature"] = x
        return output_dict, debug_info 

class softargmax_uv(torch.nn.Module):
    def __init__(self, dim, down_scale, num_vec):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, num_vec, 1, 1, 0)
        self.softargmax = SoftArgmax2D()
        
    def forward(self,x,y):
        """
        output -> downscaled uv coordinate
        """
        output_dict = {}
        debug_info = {}
        x = self.conv(x)
        B,C,H,W = x.shape
        uv_feature, _, smax = self.softargmax(x)

        output_dict["key_coords"] = uv_feature
        if self.training == False:
            debug_info["extractor_heatmap"] = smax.detach().cpu()
            debug_info["extractor_uv"] = (uv_feature * self.down_scale).detach().cpu()
        return output_dict, debug_info
    
class softargmax_feature(nn.Module):
    def __init__(self,dim,down_scale,num_vec=1,softmax_temp=1.0):
        super().__init__()
        
        self.num_vec = num_vec
        self.mask_conv = nn.Conv2d(dim, self.num_vec, 1, 1, 0)
        self.softmax = torch.nn.Softmax(dim=3)
        self.softmax_temp = Parameter(torch.ones(1)*softmax_temp)
            
        self.down_scale = down_scale
            
    def forward(self,x,y):
        """
        input 
        x: feature B,C,H,W
        y: pose B,N,2
        """
        debug_info = {}
        output_dict = {}
        B,C,H,W = x.shape
        device = x.device
        
        # get image key feature
        mask = self.mask_conv(x) # B, K, H, W
        mask = self.softmax_2d(mask, self.softmax_temp) # B, K, H, W
        x = self.summarize(mask, x) # B, K, C
        norm_coords, coords = self.get_coords(mask)
        output_dict["key_feature"] = x
        output_dict["key_coords"] = norm_coords
        
        if self.training == False:
            debug_info["extractor_uv"] = (coords * self.down_scale).detach().cpu()
            debug_info["extractor_heatmap"] = mask.detach().cpu()
        return output_dict, debug_info
        
    def summarize(self,mask,feature):
        B,C,H,W = feature.shape
        mask_c = repeat(mask, 'B K H W -> B C K H W',C=C)
        feature_c = repeat(feature, 'B C H W -> B C K H W', K=self.num_vec)
        feature_c = feature_c * mask_c
        feature_c = rearrange(feature_c, 'B C K H W -> B K C (H W)')
        feature_c = torch.sum(feature_c, 3)
        
        return feature_c
        
    def softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_softmax = torch.nn.functional.softmax(x_flat, dim=2)
        return x_softmax.view((B, C, W, H))
    
    def get_coords(self, mask):
        B, C, H, W = mask.shape
        x_channel, y_channel = self.create_coordmap(mask)
        x_value = x_channel * mask
        y_value = y_channel * mask
        
        x_value = rearrange(x_value, 'B C H W -> B C (H W)')
        y_value = rearrange(y_value, 'B C H W -> B C (H W)')
        
        x = torch.sum(x_value, 2, keepdim=True)
        y = torch.sum(y_value, 2, keepdim=True)

        norm_x = (x / H * 2) - 1
        norm_y = (x / W * 2) - 1
            
        return torch.cat([norm_x, norm_y], 2), torch.cat([x , y], 2)
    
    @torch.no_grad()
    def create_coordmap(self,x):
        B, C, H, W = x.shape
        device = x.device
        xx_ones = torch.arange(W, dtype=torch.int32).to(device)
        xx_channel = repeat(xx_ones, 'W -> B C H W', B=B,C=C,H=H)

        yy_ones = torch.arange(H, dtype=torch.int32).to(device)
        yy_channel = repeat(yy_ones, 'H -> B C H W', B=B,C=C,W=W)

        xx_channel = xx_channel.float()
        yy_channel = yy_channel.float()

        return xx_channel, yy_channel