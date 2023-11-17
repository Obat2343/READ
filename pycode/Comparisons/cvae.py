import torch
import random
import timm
import numpy as np

from torch.nn.parameter import Parameter
from einops import rearrange, repeat

from ..model.base_module import LinearBlock, Query_emb_model, StepEncoding, PositionalEncoding
from ..model.resnet_module import Resnet_Like_Decoder, Resnet_Like_Encoder

########### Model ###########

class CVAE(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_frames, num_classes=1,
                img_size=256, input_dim=4, dims=[96, 192, 384, 768], 
                enc_depths=[3,3,9,3], enc_layers=['conv','conv','conv','conv'], enc_act="gelu", enc_norm="layer", 
                dec_depths=[3,3,3], dec_layers=['conv','conv','conv'], dec_act="gelu", dec_norm="layer", drop_path_rate=0.1,
                latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0., activation="gelu", **kargs):

        super().__init__()
        
        # image encoder (UNet)
        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=latent_dim, activation=dec_act, norm=dec_norm)
        self.pool = torch.nn.AvgPool2d(16, stride=16)

        # image and motion to z
        self.encoder = Encoder_TRANSFORMER(query_keys, query_dims, num_classes=num_classes,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        # image and latent
        self.decoder = Transformer_Predictor(query_keys, query_dims, latent_dim, dropout=dropout, act=activation, img_pos_emb=True, img_guidance_rate=1.0)
        
        self.latent_dim = latent_dim

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
        reshaped_img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        return reshaped_img_feature

    def forward(self, image, query):
        img_feature = self.get_img_feature(image)

        mu, logvar = self.encoder(img_feature, query)

        z = self.reparameterize(mu, logvar)
        pred_dict = self.decoder(img_feature, z)

        return pred_dict, z, mu, logvar
        
    def reparameterize(self, mu, logvar, seed=None):
        device = mu.device
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def sample(self, image, device="cuda"):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(1, self.latent_dim).to(device)
        img_feature = self.get_img_feature(image)
        pred_dict = self.decoder(img_feature, z)

        return pred_dict

class CVAE_resnet(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_frames, num_classes=1,
                img_size=256, input_dim=4, model_name='resnet18', predictor='transformer', pretrained=False,
                latent_dim=256, img_guidance_rate=1.0, ff_size=1024, num_layers=4, num_heads=4, dropout=0., activation="gelu", **kargs):

        super().__init__()
        
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
        self.mlp = LinearBlock(feature_dim, latent_dim, activation='none')

        # image and motion to z
        self.encoder = Encoder_TRANSFORMER(query_keys, query_dims, num_classes=num_classes,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        # image and latent
        if predictor == "transformer":
            self.decoder = Transformer_Predictor(query_keys, query_dims, latent_dim, dropout=dropout, act=activation, img_pos_emb=True, img_guidance_rate=1.0)
        elif predictor == "mlp":
            self.decoder = MLP_Predictor(query_keys, query_dims, latent_dim)
        else:
            raise NotImplementedError()
        
        self.latent_dim = latent_dim

    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.backbone(img)
        return img_feature

    def forward(self, image, query):
        # encode image
        img_feature = self.get_img_feature(image)

        # extract important features from image features
        if self.rearrange_flag:
            img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        else:
            img_feature = rearrange(img_feature, "B D C -> D B C")

        img_feature = self.mlp(img_feature)

        mu, logvar = self.encoder(img_feature, query)

        z = self.reparameterize(mu, logvar)
        pred_dict = self.decoder(img_feature, z)

        return pred_dict, z, mu, logvar
        
    def reparameterize(self, mu, logvar, seed=None):
        device = mu.device
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def sample(self, image, device="cuda"):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(1, self.latent_dim).to(device)
        img_feature = self.get_img_feature(image)
        # extract important features from image features
        if self.rearrange_flag:
            img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        else:
            img_feature = rearrange(img_feature, "B D C -> D B C")

        img_feature = self.mlp(img_feature)

        pred_dict = self.decoder(img_feature, z)

        return pred_dict

class Encoder_TRANSFORMER(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_classes=1,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                    activation="gelu", **kargs):
        super().__init__()
        
        self.query_keys = query_keys
        self.query_dims = query_dims
        self.num_classes = num_classes
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        self.muQuery = torch.nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.sigmaQuery = torch.nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.query_emb_model = Query_emb_model(query_keys, query_dims, latent_dim)
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.seqTransEncoder = torch.nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        self.Linear_mu = LinearBlock(latent_dim, latent_dim, activation="none")
        self.Linear_logvar = LinearBlock(latent_dim, latent_dim, activation="none")

    def forward(self, img_feature, query):
        emb_vec = self.query_emb_model(query)
        bs, nframes, nfeats = emb_vec.shape
        emb_vec = rearrange(emb_vec, "B S D -> S B D")

        # adding the mu and sigma queries
        index = [0] * bs
        xseq = torch.cat((self.muQuery[index][None], self.sigmaQuery[index][None], emb_vec, img_feature), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        final = self.seqTransEncoder(xseq)
        mu = self.Linear_mu(final[0])
        logvar = self.Linear_logvar(final[1])
            
        return mu, logvar

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

    def forward(self, img_feature, z):
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
            cat_feature = torch.cat([query_feature, img_feature, z[None]]) # 2S B D
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
                            LinearBlock(self.feature_dim + latent_dim, int(self.feature_dim / 2), activation=activation),
                            LinearBlock(int(self.feature_dim / 2), dim * sequence_length, activation="none"))
        self.output_module_dict = torch.nn.ModuleDict(module_dict)

    def forward(self, img_feature, z):
        batch = img_feature.shape[1]
        img_feature = rearrange(img_feature, "S B D -> B (S D)")
        cat_feature = torch.cat([img_feature, z], 1)
        
        pred_dict = {}
        for key in self.output_module_dict.keys():
            motion = self.output_module_dict[key](cat_feature)
            pred_dict[key] = rearrange(motion, "B (S D) -> B S D", S=self.sequence_length)
        
        return pred_dict

# class Decoder_TRANSFORMER(torch.nn.Module):
#     def __init__(self, query_keys, query_dims, num_frames, num_classes=1,
#                     latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
#                     ablation=None, **kargs):
#         super().__init__()

#         self.num_frames = num_frames
#         self.num_classes = num_classes
        
#         self.latent_dim = latent_dim
        
#         self.ff_size = ff_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout

#         self.activation = activation
#         self.actionBiases = torch.nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
#         self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
#         seqTransDecoderLayer = torch.nn.TransformerDecoderLayer(d_model=self.latent_dim,
#                                                         nhead=self.num_heads,
#                                                         dim_feedforward=self.ff_size,
#                                                         dropout=self.dropout,
#                                                         activation=activation)
#         self.seqTransDecoder = torch.nn.TransformerDecoder(seqTransDecoderLayer,
#                                                         num_layers=self.num_layers)
        
#         module_dict = {}
#         for key, dim in zip(query_keys, query_dims):
#             if key == "time":
#                 continue
#             module_dict[key] = torch.nn.Sequential(
#                             LinearBlock(latent_dim, int(latent_dim / 2)),
#                             LinearBlock(int(latent_dim / 2), dim, activation="none"))
        
#         self.output_module_dict = torch.nn.ModuleDict(module_dict)
        
#     def forward(self, image_feature, z):
#         bs, latent_dim = z.shape
#         nframes = self.num_frames
        
#         # shift the latent noise vector to be the action noise
#         z = z + self.actionBiases[0]
#         z = z[None]  # sequence of size 1
#         z = torch.cat([z, image_feature], 0)
            
#         timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
#         timequeries = self.sequence_pos_encoder(timequeries)
        
#         output = self.seqTransDecoder(tgt=timequeries, memory=z)
        
# #         output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
#         output = rearrange(output, "S B D -> B S D")
        
#         pred_dict = {}
#         for key in self.output_module_dict.keys():
#             pred_dict[key] = self.output_module_dict[key](output)
        
#         return pred_dict

############# Loss ###############

class VAE_Loss(torch.nn.Module):
    
    def __init__(self, rot_mode="6d", kld_weight=0.1, device="cuda"):
        super().__init__()
        self.motion_loss = Motion_Loss(device=device, mode=rot_mode)
        self.kld_weight = kld_weight
        
    def forward(self, pred_dict, gt_dict, mu, log_var, mode="train"):
        
        # Motion Loss
        loss, loss_dict = self.motion_loss(pred_dict, gt_dict, mode=mode)
        
        # KLD loss
        if mu.dim() == 2:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        elif mu.dim() == 3:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 2))
        else:
            raise ValueError()

        # sum
        loss += self.kld_weight * kld_loss
        
        # register loss value to dict for weights and biases
        loss_dict[f"{mode}/KLD"] = kld_loss.item()
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
