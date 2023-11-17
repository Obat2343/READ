import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from torch.nn.parameter import Parameter
from timm.models.layers import DropPath

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



##### Conv Block #####

class ConvLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel, stride, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel, stride, pad)
        
    def forward(self,x):
        return self.conv(x), {}

class LinearBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, activation='prelu', norm=None):
        super(LinearBlock, self).__init__()

        self.linear = torch.nn.Linear(input_size, output_size)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(False)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, False)
        elif self.activation == 'gelu':
            self.act = torch.nn.GELU()
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'none':
            self.activation = None
    
    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.linear(x))
        else:
            out = self.linear(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Transformer_Block(torch.nn.Module):
    
    def __init__(self, dim, attn_block=4, head=4, act="gelu", norm="none"):
        super().__init__()
        
        self.emb_dim = dim
        self.head = head
        self.num_attn_block = attn_block
        
        qkv_list = []
        attn_list = []
        ff_list = []
        for _ in range(self.num_attn_block):
            qkv_list.append(LinearBlock(self.emb_dim, self.emb_dim*3, activation=act, norm=norm))
            attn_list.append(torch.nn.MultiheadAttention(self.emb_dim, head, batch_first=True))
            ff_list.append(torch.nn.Sequential(
                            LinearBlock(self.emb_dim, self.emb_dim, activation=act, norm=norm),
                            LinearBlock(self.emb_dim, self.emb_dim, activation=act, norm=norm),
                            LinearBlock(self.emb_dim, self.emb_dim, activation=act, norm=norm)))
        
        self.qkv_modules = torch.nn.ModuleList(qkv_list)
        self.attn_modules = torch.nn.ModuleList(attn_list)
        self.ff_modules = torch.nn.ModuleList(ff_list)
        
    def forward(self, emb_vec):
        """
        Input and Output:
        emb_vec: torch.tensor -> shape: (B, S, D) Bacth, Length of Sequence, Dim of feature
        """
                
        for qkv_module, attn_module, ff_module in zip(self.qkv_modules, self.attn_modules, self.ff_modules):
            
            qkv = qkv_module(emb_vec)
            q, k, v = qkv[:,:,:self.emb_dim], qkv[:,:,self.emb_dim:2*self.emb_dim], qkv[:,:,self.emb_dim*2:]

            attn_emb, attn_weights = attn_module(q,k,v)
            attn_emb = torch.nan_to_num(attn_emb, nan=0.0)
        
            emb_vec = emb_vec + attn_emb
            emb_vec = emb_vec + ff_module(emb_vec)
        
        return emb_vec

class ConvBlock(nn.Module):
    r""" 
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel=3, stride=1, padding=1, act='gelu', norm='layer', drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim // 4, 1, 1, 0)
        self.conv1 = nn.Conv2d(dim // 4, dim // 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim // 4, dim, 1, 1, 0)
        self.norm = self.norm_layer(dim // 4, name=norm)
        self.act = self.activation_layer(act)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv0(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x, None
    
    @staticmethod
    def norm_layer(dim, name='none',num_group=0):
        if name == 'batch':
            layer = nn.BatchNorm2d(dim)
        elif name == 'layer':
            layer = nn.GroupNorm(1, dim)
        elif name == 'instance':
            layer = nn.InstanceNorm2d(dim)
        elif name == 'group':
            assert num_group == 0, "change num_group. Current num_group is 0"
            layer = nn.GroupNorm(num_group, dim)
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid norm")
        return layer
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = nn.ReLU()
        elif name == 'prelu':
            layer = nn.PReLU()
        elif name == 'lrelu':
            layer = nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = nn.Tanh()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'gelu':
            layer = nn.GELU()
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x, None

class CoordConv2d(torch.nn.modules.conv.Conv2d):
    """
    from https://github.com/walsvid/CoordConv/blob/master/coordconv.py
         https://arxiv.org/pdf/1807.03247.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.with_r = with_r
        self.conv = nn.Conv2d(in_channels + 2 + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
    
    def addcoords(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        device = input_tensor.device
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        xx_channel = xx_channel.to(device)
        yy_channel = yy_channel.to(device)
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out

class SoftArgmax2D(torch.nn.Module):
    """
    https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, temp=1.0, activate="softmax", norm=True):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        if activate == 'softmax':
            self.activate = torch.nn.Softmax(dim=2)
        elif activate == 'sigmoid':
            self.activate = torch.nn.Sigmoid()
        self.temp = Parameter(torch.ones(1)*temp)
        self.norm = norm

    def _activate_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_activate = self.activate(x_flat)
        return x_activate.view((B, C, W, H))


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))
        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        device = x.device
        batch_size, channels, height, width = x.size()
        
        # comupute argmax
        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(torch.div(argmax.float(), float(width)))
        argmax = torch.cat((argmax_x.view(batch_size, channels, -1), (argmax_y.view(batch_size, channels, -1))), 2)
        
        smax = self._activate_2d(x, self.temp)

        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).to(device)
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)
        if self.norm:
            x_coords = x_coords / width

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).to(device)
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)
        if self.norm:
            y_coords = y_coords / height

        softargmax = torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)
        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))
        
        # Put the x coords and y coords (shape (B,C)) into an output with shape (B,C,2)
        return softargmax, argmax, smax

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Query_emb_model(torch.nn.Module):
    def __init__(self, query_keys, query_dims, emb_dim, act="gelu"):
        """
        Input:
        query_keys: list of query keys you want to use. Other queries will be ignored in the forward process.
        query_dims: list of dim of each query that you want to use.
        emb_dim: dimension of output feature (embedded query)
        """
        super().__init__()

        self.register_query_keys = query_keys
        query_total_dim = sum(query_dims)
        self.query_emb_model = self.make_linear_model(query_total_dim, emb_dim, act)

    def forward(self, querys):
        """
        Input
        querys: dict
            key:
                str
            value:
                torch.tensor: shape -> (B, S, D), B -> Batch Size, N, Num of query in each batch, S -> Sequence Length, D -> Dim of each values
        Output:
        query_emb: torch.tensor: shape -> (B, S, QD), QD -> emb_dim
        """
        keys = list(querys.keys())
        keys.sort()

        query_list = []
        for key in keys:
            if key in self.register_query_keys:
                D = querys[key].dim()
                query_list.append(querys[key])
        
        if len(query_list) > 1:
            query_cat = torch.cat(query_list, D-1)
        else:
            query_cat = query_list[0]
            
        query_emb = self.query_emb_model(query_cat)
        return query_emb
        
    def make_linear_model(self, input_dim, output_dim, act):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            torch.nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            torch.nn.Linear(output_dim * 2, output_dim))
        return model
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = torch.nn.ReLU()
        elif name == 'prelu':
            layer = torch.nn.PReLU()
        elif name == 'lrelu':
            layer = torch.nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = torch.nn.Tanh()
        elif name == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif name == 'gelu':
            layer = torch.nn.GELU()
        elif name == 'none':
            layer = torch.nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class StepEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.linear_layer = self.make_linear_model(dim, dim, "gelu")

    def forward(self, time):
        """
        -----------------------------
        inputs
        x: torch.tensor(S, B, D)
        time: torch.tensor(B)
        -----------------------------
        S: length of sequence
        B: Batch size
        D: Dimension of feature
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_len) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = self.linear_layer(embeddings)
        embeddings = repeat(embeddings, "B D -> S B D", S=1)
        return embeddings
    
    def make_linear_model(self, input_dim, output_dim, act):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            torch.nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            torch.nn.Linear(output_dim * 2, output_dim))
        return model
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = torch.nn.ReLU()
        elif name == 'prelu':
            layer = torch.nn.PReLU()
        elif name == 'lrelu':
            layer = torch.nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = torch.nn.Tanh()
        elif name == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif name == 'gelu':
            layer = torch.nn.GELU()
        elif name == 'none':
            layer = torch.nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)