import torch
import random
import timm

from torch.nn.parameter import Parameter

from einops import rearrange, repeat

from ..model.base_module import LinearBlock, Query_emb_model, StepEncoding, PositionalEncoding
from ..model.resnet_module import Resnet_Like_Decoder, Resnet_Like_Encoder

class Deterministic_planner(torch.nn.Module):
    def __init__(self, query_keys, query_dims, img_size=256, input_dim=4, dims=[96, 192, 384, 768], 
                    enc_depths=[3,3,9,3], enc_layers=['conv','conv','conv','conv'], enc_act="gelu", enc_norm="layer", 
                    dec_depths=[3,3,3], dec_layers=['conv','conv','conv'], dec_act="gelu", dec_norm="layer", drop_path_rate=0.1,
                    extractor_name="query_uv_feature", predictor_act="gelu",
                    predictor_drop=0., img_guidance_rate=1.0, query_emb_dim=256):
        """
        Args:
        img_size (int): Size of image. We assume image is square.
        input_dim (int): Size of channel. 3 for RGB image.
        enc_depths (list[int]): Number of blocks at each stage for encoder.
        dec_depths (list[int]): Number of blocks at each stage for decoder.
        dims (list[int]): The channel size of each feature map (depth).
        enc_layers (list[str]): Name of layer at each stage of encoder.
        dec_layers (list[str]): Name of layer at each stage of decoder.
        extractor_name (str): Name of predictor
        predictor_act (str): Activation function.
        drop_path (float): Stochastic depth rate for encoder and decoder. Default: 0.1
        """
        super().__init__()
        self.img_size = img_size
        if query_emb_dim == 0:
            emb_dim = dims[0] // 2
        else:
            emb_dim = query_emb_dim
        
        # image encoder (UNet)
        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        
        # extract important feature
        self.pool = torch.nn.AvgPool2d(16, stride=16)
        
        # predictor
        self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_pos_emb=True, img_guidance_rate=img_guidance_rate)

    def forward(self, img):
        
        img_feature = self.get_img_feature(img)
        
        # extract important features from image features
        reshaped_img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        output_dict = self.predictor(reshaped_img_feature)
            
        return output_dict

    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        img_feature = self.pool(img_feature)
        self.img_feature = img_feature
        return img_feature

class Transformer_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim, sequence_length=101,
                    ff_size=1024, num_layers=8, num_heads=4, dropout=0.0,
                    activation="gelu", img_pos_emb=True, img_guidance_rate=1.0, **kargs):
        super().__init__()
        """
        This model is based of Motion Diffusion Model from https://arxiv.org/pdf/2209.14916.pdf
        The image features is fed into the model insted of the text features.
        """
        # configuration
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.img_guidance_rate = img_guidance_rate
        self.img_pos_emb = img_pos_emb
        
        self.learnable_query = Parameter(torch.randn(sequence_length, latent_dim))
        
        # Transformer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.seqTransEncoder = torch.nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        
        # decoder
        module_dict = {}
        for key, dim in zip(query_keys, query_dims):
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), dim, activation="none"))
        self.output_module_dict = torch.nn.ModuleDict(module_dict)

    def forward(self, img_feature):
        batch = img_feature.shape[1]
        # embedding the pose to the vector
        nframes, nfeats = self.learnable_query.shape
        query_feature = repeat(self.learnable_query, "S D -> B S D", B=batch)
        query_feature = rearrange(query_feature, "B S D -> S B D")
        
        # add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature)
        if self.img_pos_emb:
            img_feature = self.sequence_pos_encoder(img_feature)
        
        if random.random() <= self.img_guidance_rate:
            cat_feature = torch.cat([query_feature, img_feature]) # 2S B D
        else:
            cat_feature = query_feature
        
        # transformer process
        motion_feature = self.seqTransEncoder(cat_feature) # 2S B D
        
        # decode
        motion_feature = rearrange(motion_feature[:nframes], "S B D -> B S D") # 2S B D -> B S D
        
        pred_noise_dict = {}
        for key in self.output_module_dict.keys():
            pred_noise_dict[key] = self.output_module_dict[key](motion_feature)
        
        return pred_noise_dict

class MLP_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim, sequence_length=101,
                    activation="gelu", **kargs):
        super().__init__()
        """
        This model is based of Motion Diffusion Model from https://arxiv.org/pdf/2209.14916.pdf
        The image features is fed into the model insted of the text features.
        """
        # configuration
        self.latent_dim = latent_dim
        self.activation = activation
        self.feature_dim = latent_dim * 8 * 8
        self.sequence_length = sequence_length
        
        # decoder
        module_dict = {}
        for key, dim in zip(query_keys, query_dims):
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(self.feature_dim, int(self.feature_dim / 2), activation=activation),
                            LinearBlock(int(self.feature_dim / 2), dim * sequence_length, activation="none"))
        self.output_module_dict = torch.nn.ModuleDict(module_dict)

    def forward(self, img_feature):
        batch = img_feature.shape[1]
        img_feature = rearrange(img_feature, "S B D -> B (S D)")
        
        pred_dict = {}
        for key in self.output_module_dict.keys():
            motion = self.output_module_dict[key](img_feature)
            pred_dict[key] = rearrange(motion, "B (S D) -> B S D", S=self.sequence_length)
        
        return pred_dict

class Deterministic_planner_Resnet(torch.nn.Module):
    def __init__(self, query_keys, query_dims, img_size=256, input_dim=4, predictor_act="gelu", predictor='transformer',
                    predictor_drop=0., img_guidance_rate=1.0, query_emb_dim=256, pretrained=False, model_name='resnet18'):
        """
        Args:
        img_size (int): Size of image. We assume image is square.
        input_dim (int): Size of channel. 3 for RGB image.
        enc_depths (list[int]): Number of blocks at each stage for encoder.
        dec_depths (list[int]): Number of blocks at each stage for decoder.
        dims (list[int]): The channel size of each feature map (depth).
        enc_layers (list[str]): Name of layer at each stage of encoder.
        dec_layers (list[str]): Name of layer at each stage of decoder.
        extractor_name (str): Name of predictor
        predictor_act (str): Activation function.
        drop_path (float): Stochastic depth rate for encoder and decoder. Default: 0.1
        """
        super().__init__()
        self.img_size = img_size
        if query_emb_dim == 0:
            emb_dim = dims[0] // 2
        else:
            emb_dim = query_emb_dim
        
        if 'resnet' in model_name:
            self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=input_dim)
            self.rearrange_flag = True
        elif 'vit' in model_name:
            self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=input_dim, img_size=img_size)
            self.rearrange_flag = False
        else:
            raise NotImplementedError()
        
        self.backbone.reset_classifier(num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features

        # change dim
        self.mlp = LinearBlock(feature_dim, emb_dim, activation='none')

        # predictor
        if predictor == "transformer":
            self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_pos_emb=True, img_guidance_rate=img_guidance_rate)
        elif predictor == "mlp":
            self.predictor = MLP_Predictor(query_keys, query_dims, emb_dim)
        else:
            raise NotImplementedError()

    def forward(self, img):
        # encode image
        img_feature = self.get_img_feature(img)

        # extract important features from image features
        if self.rearrange_flag:
            img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        else:
            img_feature = rearrange(img_feature, "B D C -> D B C")

        img_feature = self.mlp(img_feature)
        
        # extract important features from image features
        output_dict = self.predictor(img_feature)
            
        return output_dict

    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.backbone(img)
        return img_feature

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
        loss = self.MSE(pred_rot, gt_dict["rotation"].to(self.device))
        return loss

class UV_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(UV_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_uv = pred_dict["uv"].to(self.device)
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
        gt_grasp = gt_dict["grasp_state"].to(self.device)
        
        loss = self.MSE(pred_grasp, gt_grasp)
        return loss 
