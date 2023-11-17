import abc
import random
import torch
import timm

from einops import rearrange

from ..model.base_module import LinearBlock, Query_emb_model, StepEncoding, PositionalEncoding
from ..model.resnet_module import Resnet_Like_Decoder, Resnet_Like_Encoder

class MLP_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim=32, num_layers=4,
                    activation="gelu", **kargs):
        super().__init__()
        """
        This model is for Toy datasets.
        """
        # configuration
        self.register_query_keys = query_keys
        self.register_query_keys.sort()

        query_total_dim = sum(query_dims) + 1
        
        module_list = [LinearBlock(query_total_dim, latent_dim, activation=activation)]
        for i in range(num_layers-1):
            module_list.append(LinearBlock(latent_dim, latent_dim, activation=activation))
        self.feature_extractor = torch.nn.Sequential(*module_list)

        # decoder
        module_dict = {}
        for key, dim in zip(query_keys, query_dims):
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), dim, activation="none"))
        self.output_module_dict = torch.nn.ModuleDict(module_dict)
    
    def forward(self, query, time_step):
        """
        Input
            query: dict
            time_step: (B, 1)
        Output:
            noise_dict: dict
        """

        query_list = []
        for key in self.register_query_keys:
            query_list.append(query[key])
        query_list.append(torch.unsqueeze(time_step, 1))
        query_cat = torch.cat(query_list, 1)

        feature = self.feature_extractor(query_cat)

        pred_noise_dict = {}
        for key in self.output_module_dict.keys():
            pred_noise_dict[key] = query[key] + self.output_module_dict[key](feature)
        
        return pred_noise_dict

class MLP_Energy_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim=32, num_layers=4,
                    activation="gelu", **kargs):
        super().__init__()
        """
        This model is for Toy datasets.
        """
        # configuration
        self.register_query_keys = query_keys
        self.register_query_keys.sort()

        query_total_dim = sum(query_dims) + 1
        
        module_list = [LinearBlock(query_total_dim, latent_dim, activation=activation)]
        for i in range(num_layers-1):
            module_list.append(LinearBlock(latent_dim, latent_dim, activation=activation))
        self.feature_extractor = torch.nn.Sequential(*module_list)

        # decoder
        self.output_module = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), 1, activation="none"))
    
    def forward(self, query, time_step):
        """
        Input
            query: dict
            time_step: (B, 1)
        Output:
            noise_dict: dict
        """

        query_list = []
        for key in self.register_query_keys:
            query_list.append(query[key])
        query_list.append(torch.unsqueeze(time_step, 1))
        query_cat = torch.cat(query_list, 1)

        feature = self.feature_extractor(query_cat)

        pred_energy = self.output_module(feature)
        
        return pred_energy

class Abstract_Continuous_Diffusion(abc.ABC, torch.nn.Module):

    @abc.abstractmethod
    def forward(self, img, query, time_step, with_feature=False):
        pass

    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        self.img_feature = img_feature
        return img_feature
    
    def set_condition(self, img):
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        self.img_feature = img_feature

    def get_extracted_img_feature(self, img, query, with_feature=False):
        device = img.device
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        uv = query['uv'].to(device)
        img_feature_dict, _ = self.ife(img_feature, uv) # TODO: dict?
        
        return img_feature_dict["img_feature"]

class SPE_Continuous_Diffusion(Abstract_Continuous_Diffusion):
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
        self.ife = Image_feature_extractor_model(extractor_name, emb_dim, 1)
        
        # predictor
        self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)

    def forward(self, img, query, time_step, with_feature=False):
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        img_feature_dict, _ = self.ife(img_feature, query['uv'])
        
        output_dict = self.predictor(img_feature_dict["img_feature"], query, time_step)
            
        return output_dict

class SPE_Continuous_Diffusion_CG(Abstract_Continuous_Diffusion):
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
        self.ife = Image_feature_extractor_model(extractor_name, emb_dim, 1)
        
        # predictor
        self.predictor = Transformer_Predictor_with_retrieval_guidance(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)

    def forward(self, img, query, retrieved_query, time_step, with_feature=False):
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        img_feature_dict, _ = self.ife(img_feature, query['uv'])
        
        output_dict = self.predictor(img_feature_dict["img_feature"], query, retrieved_query, time_step)
            
        return output_dict
    
    def classifier_free_guidance(self, img, query, retrieved_query, time_step, guidance_rate=0.9, with_feature=False):

        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        img_feature_dict, _ = self.ife(img_feature, query['uv'])
        
        output_dict = self.predictor.guided_prediction(img_feature_dict["img_feature"], query, retrieved_query, time_step, guidance_rate)
            
        return output_dict

class SPE_Continuous_Latent_Diffusion(Abstract_Continuous_Diffusion):
    def __init__(self, query_keys, query_dims, encoder, latent_dim, inout_type,
                    img_size=256, input_dim=4, dims=[96, 192, 384, 768], 
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
        
        self.inout_type = inout_type
        
        # image encoder (UNet)
        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        
        # extract important feature
        self.ife = Image_feature_extractor_model(extractor_name, emb_dim, 1)
        
        # predictor
        if self.inout_type in ["l-m-l", "l-m"]:
            self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)
        else:
            self.predictor = Latent_Transformer_Predictor(latent_dim, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)
        
        # latent encoder
        self.latent_AutoEncoder = encoder
        for param in self.latent_AutoEncoder.parameters():
            param.requires_grad = False

    def forward(self, img, latent, time_step, with_feature=False):
        """
        Args:
        img (tensor): an image, shape (B, C , H, W)
        latent (tensor): represent the motion as a latent vector, shape (B, D)
        time_step (tensor): diffusion time-step (B, 1)?
        with_feature (bool): if True, use self.img_feature instead of input img. This is for reducing cost when the same image is used.
        """
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        with torch.no_grad():
            query = self.latent_AutoEncoder.decode(latent)

        # extract important features from image features
        img_feature_dict, _ = self.ife(img_feature, query['uv'])
        
        if self.inout_type == "l":
            return self.predictor(img_feature_dict["img_feature"], latent, time_step)
        elif self.inout_type == "l-m":
            return self.predictor(img_feature_dict["img_feature"], query, time_step)
        elif self.inout_type == "l-m-l":
            pred_query = self.predictor(img_feature_dict["img_feature"], query, time_step)
            pred_latent = self.latent_AutoEncoder.encode(pred_query)
            for key in latent.keys():
                latent[key] = latent[key] - pred_latent[key]
            return latent
        else:
            raise ValueError(f"{self.inout_type} is not available")

    def get_extracted_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        # extract important features from image features
        img_feature_dict, _ = self.ife(img_feature, query['uv'])
        return img_feature_dict["img_feature"]

class AvgPool_Continuous_Diffusion(Abstract_Continuous_Diffusion):
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

    def forward(self, img, query, time_step, with_feature=False):
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        reshaped_img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        output_dict = self.predictor(reshaped_img_feature, query, time_step)
            
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
    
    def set_condition(self, img):
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        img_feature = self.pool(img_feature)
        self.img_feature = img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        return img_feature

class AvgPool_Continuous_Latent_Diffusion(Abstract_Continuous_Diffusion):
    def __init__(self, query_keys, query_dims, encoder, latent_dim, inout_type,
                    img_size=256, input_dim=4, dims=[96, 192, 384, 768], 
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
        
        self.inout_type = inout_type
        
        # image encoder (UNet)
        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        
        # extract important feature
        self.pool = torch.nn.AvgPool2d(16, stride=16)
        
        # predictor
        if self.inout_type in ["l-m-l", "l-m"]:
            self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_pos_emb=True, img_guidance_rate=img_guidance_rate)
        else:
            self.predictor = Latent_Transformer_Predictor(latent_dim, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)
        
        # latent encoder
        self.latent_AutoEncoder = encoder
        for param in self.latent_AutoEncoder.parameters():
            param.requires_grad = False

    def forward(self, img, latent, time_step, with_feature=False):
        """
        Args:
        img (tensor): an image, shape (B, C , H, W)
        latent (tensor): represent the motion as a latent vector, shape (B, D)
        time_step (tensor): diffusion time-step (B, 1)?
        with_feature (bool): if True, use self.img_feature instead of input img. This is for reducing cost when the same image is used.
        """
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        with torch.no_grad():
            query = self.latent_AutoEncoder.decode(latent)

        # extract important features from image features
        reshaped_img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        
        if self.inout_type == "l":
            return self.predictor(reshaped_img_feature, latent, time_step)
        elif self.inout_type == "l-m":
            return self.predictor(reshaped_img_feature, query, time_step)
        elif self.inout_type == "l-m-l":
            pred_query = self.predictor(reshaped_img_feature, query, time_step)
            pred_latent = self.latent_AutoEncoder.encode(pred_query)
            for key in latent.keys():
                latent[key] = latent[key] - pred_latent[key]
            return latent
        else:
            raise ValueError(f"{self.inout_type} is not available")

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
    
    def set_condition(self, img):
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        img_feature = self.pool(img_feature)
        self.img_feature = img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        return img_feature

class ConvPool_Continuous_Diffusion(Abstract_Continuous_Diffusion):
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
        self.pool = torch.nn.Sequential(torch.nn.Conv2d(emb_dim, emb_dim, 4, stride=4, padding=0),
                                        torch.nn.Conv2d(emb_dim, emb_dim, 4, stride=4, padding=0))
        
        # predictor
        self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_pos_emb=True, img_guidance_rate=img_guidance_rate)

    def forward(self, img, query, time_step, with_feature=False):
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        reshaped_img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        output_dict = self.predictor(reshaped_img_feature, query, time_step)
            
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
    
    def set_condition(self, img):
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        img_feature = self.pool(img_feature)
        self.img_feature = img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        return img_feature

class ConvPool_Continuous_Latent_Diffusion(Abstract_Continuous_Diffusion):
    def __init__(self, query_keys, query_dims, encoder, latent_dim, inout_type,
                    img_size=256, input_dim=4, dims=[96, 192, 384, 768], 
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
        
        self.inout_type = inout_type
        
        # image encoder (UNet)
        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        
        # extract important feature
        self.pool = torch.nn.Sequential(torch.nn.Conv2d(emb_dim, emb_dim, 4, stride=4, padding=0),
                                        torch.nn.Conv2d(emb_dim, emb_dim, 4, stride=4, padding=0))
        
        # predictor
        if self.inout_type in ["l-m-l", "l-m"]:
            self.predictor = Transformer_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_pos_emb=True, img_guidance_rate=img_guidance_rate)
        else:
            self.predictor = Latent_Transformer_Predictor(latent_dim, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)
        
        # latent encoder
        self.latent_AutoEncoder = encoder
        for param in self.latent_AutoEncoder.parameters():
            param.requires_grad = False

    def forward(self, img, latent, time_step, with_feature=False):
        """
        Args:
        img (tensor): an image, shape (B, C , H, W)
        latent (tensor): represent the motion as a latent vector, shape (B, D)
        time_step (tensor): diffusion time-step (B, 1)?
        with_feature (bool): if True, use self.img_feature instead of input img. This is for reducing cost when the same image is used.
        """
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        with torch.no_grad():
            query = self.latent_AutoEncoder.decode(latent)

        # extract important features from image features
        reshaped_img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        
        if self.inout_type == "l":
            return self.predictor(reshaped_img_feature, latent, time_step)
        elif self.inout_type == "l-m":
            return self.predictor(reshaped_img_feature, query, time_step)
        elif self.inout_type == "l-m-l":
            pred_query = self.predictor(reshaped_img_feature, query, time_step)
            pred_latent = self.latent_AutoEncoder.encode(pred_query)
            for key in latent.keys():
                latent[key] = latent[key] - pred_latent[key]
            return latent
        else:
            raise ValueError(f"{self.inout_type} is not available")

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
    
    def set_condition(self, img):
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        img_feature = self.pool(img_feature)
        self.img_feature = img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        return img_feature

class Timm_Continuous_Diffusion(Abstract_Continuous_Diffusion):

    def __init__(self, query_keys, query_dims, model_name, img_size=256, input_dim=4, pretrained=True,
                predictor_act="gelu", predictor_drop=0., img_guidance_rate=1.0, query_emb_dim=256):
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
        self.mlp = LinearBlock(feature_dim, query_emb_dim, activation='none')

        # predictor
        self.predictor = Transformer_Predictor(query_keys, query_dims, query_emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)

    def forward(self, img, query, time_step, with_feature=False):
        """
        Args:
        img (tensor): an image, shape (B, C , H, W)
        latent (tensor): represent the motion as a latent vector, shape (B, D)
        time_step (tensor): diffusion time-step (B, 1)?
        with_feature (bool): if True, use self.img_feature instead of input img. This is for reducing cost when the same image is used.
        """
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        # extract important features from image features
        if self.rearrange_flag:
            img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        else:
            img_feature = rearrange(img_feature, "B D C -> D B C")

        img_feature = self.mlp(img_feature)

        output_dict = self.predictor(img_feature, query, time_step)
        return output_dict
    
    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.backbone(img)
        self.img_feature = img_feature
        return img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        return img_feature
    
    def set_condition(self, img):
        img_feature = self.backbone(img)
        self.img_feature = img_feature

class Timm_Continuous_Latent_Diffusion(Abstract_Continuous_Diffusion):

    def __init__(self, query_keys, query_dims, model_name, encoder, latent_dim, inout_type, img_size=256, input_dim=4, pretrained=True,
                predictor_act="gelu", predictor_drop=0., img_guidance_rate=1.0, query_emb_dim=256):
        
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
        self.mlp = LinearBlock(feature_dim, query_emb_dim, activation='none')

        # predictor
        self.inout_type = inout_type
        if self.inout_type in ["l-m-l", "l-m"]:
            if self.inout_type == "l-m-l":
                raise NotImplementedError("l-m-l possibly works but not be debugged.")
            self.predictor = Transformer_Predictor(query_keys, query_dims, query_emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)
        else:
            self.predictor = Latent_Transformer_Predictor(latent_dim, query_emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)

        # latent encoder
        self.latent_AutoEncoder = encoder
        for param in self.latent_AutoEncoder.parameters():
            param.requires_grad = False

    def forward(self, img, latent, time_step, with_feature=False):
        """
        Args:
        img (tensor): an image, shape (B, C , H, W)
        latent (tensor): represent the motion as a latent vector, shape (B, D)
        time_step (tensor): diffusion time-step (B, 1)?
        with_feature (bool): if True, use self.img_feature instead of input img. This is for reducing cost when the same image is used.
        """
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        with torch.no_grad():
            query = self.latent_AutoEncoder.decode(latent)

        # extract important features from image features
        if self.rearrange_flag:
            img_feature = rearrange(img_feature, "B C H W -> (H W) B C")
        else:
            img_feature = rearrange(img_feature, "B D C -> D B C")

        img_feature = self.mlp(img_feature)

        if self.inout_type == "l":
            return self.predictor(img_feature, latent, time_step)
        elif self.inout_type == "l-m":
            return self.predictor(img_feature, query, time_step)
        elif self.inout_type == "l-m-l":
            pred_query = self.predictor(img_feature, query, time_step)
            pred_latent = self.latent_AutoEncoder.encode(pred_query)
            for key in latent.keys():
                latent[key] = latent[key] - pred_latent[key]
            return latent
        else:
            raise ValueError(f"{self.inout_type} is not available")
        
    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.backbone(img)
        self.img_feature = img_feature
        return img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature

        return img_feature
    
    def set_condition(self, img):
        img_feature = self.backbone(img)
        self.img_feature = img_feature

class Transformer_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim, 
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
        
        # embedding pose to vector
        self.query_emb_model = Query_emb_model(query_keys, query_dims, latent_dim)
        
        # embedding noise step
        self.steps_encoder = StepEncoding(latent_dim)
        
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

    def forward(self, img_feature, query, time_step):
        # embedding the pose to the vector
        query_feature = self.query_emb_model(query)
        bs, nframes, nfeats = query_feature.shape
        query_feature = rearrange(query_feature, "B S D -> S B D")
        
        # add time embedding 
        time_emb = self.steps_encoder(time_step)
        query_feature = torch.cat([query_feature, time_emb], 0) # (S+1) B D
        
        # add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature)
        if self.img_pos_emb:
            img_feature = self.sequence_pos_encoder(img_feature)
        
        if random.random() <= self.img_guidance_rate:
            cat_feature = torch.cat([query_feature, img_feature]) # (2S + 1) B D
        else:
            cat_feature = query_feature
        
        # transformer process
        motion_feature = self.seqTransEncoder(cat_feature) # (2S + 1) B D
        
        # decode
        motion_feature = rearrange(motion_feature[:nframes], "S B D -> B S D") # (2S + 1) B D -> B S D
        
        pred_noise_dict = {}
        for key in self.output_module_dict.keys():
            pred_noise_dict[key] = query[key] + self.output_module_dict[key](motion_feature)
        
        return pred_noise_dict

class Transformer_Predictor_with_retrieval_guidance(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim, 
                    ff_size=1024, num_layers=8, num_heads=4, dropout=0.0,
                    activation="gelu", img_pos_emb=True, img_guidance_rate=1.0, retrieval_guidance_rate=0.9, **kargs):
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
        self.retrieval_guidance_rate = 0.9
        self.img_pos_emb = img_pos_emb
        
        # embedding pose to vector
        self.query_emb_model = Query_emb_model(query_keys, query_dims, latent_dim)
        self.retrieval_emb_model = Query_emb_model(query_keys, query_dims, latent_dim)
        
        # embedding noise step
        self.steps_encoder = StepEncoding(latent_dim)
        
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

    def forward(self, img_feature, query, retrieved_query, time_step):
        # embedding the pose to the vector
        query_feature = self.query_emb_model(query)
        retrieval_feature = self.retrieval_emb_model(retrieved_query)

        bs, nframes, nfeats = query_feature.shape
        query_feature = rearrange(query_feature, "B S D -> S B D")
        retrieval_feature = rearrange(retrieval_feature, "B S D -> S B D")
        
        # add time embedding 
        time_emb = self.steps_encoder(time_step)
        query_feature = torch.cat([query_feature, time_emb], 0) # (S+1) B D
        
        # add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature)
        if self.img_pos_emb:
            img_feature = self.sequence_pos_encoder(img_feature)
        
        retrieved_feature = self.sequence_pos_encoder(retrieval_feature)
        
        if random.random() <= self.img_guidance_rate:
            query_feature = torch.cat([query_feature, img_feature]) # (2S + 1) B D
        if random.random() <= self.retrieval_guidance_rate:
            query_feature = torch.cat([query_feature, retrieval_feature])
        
        # transformer process
        motion_feature = self.seqTransEncoder(query_feature) # (2S + 1) B D
        
        # decode
        motion_feature = rearrange(motion_feature[:nframes], "S B D -> B S D") # (2S + 1) B D -> B S D
        
        pred_noise_dict = {}
        for key in self.output_module_dict.keys():
            pred_noise_dict[key] = query[key] + self.output_module_dict[key](motion_feature)
        
        return pred_noise_dict
    
    def guided_prediction(self, img_feature, query, retrieved_query, time_step, guidance_rate=0.8):
        # embedding the pose to the vector
        query_feature = self.query_emb_model(query)
        retrieval_feature = self.retrieval_emb_model(retrieved_query)

        bs, nframes, nfeats = query_feature.shape
        query_feature = rearrange(query_feature, "B S D -> S B D")
        retrieval_feature = rearrange(retrieval_feature, "B S D -> S B D")
        
        # add time embedding 
        time_emb = self.steps_encoder(time_step)
        query_feature = torch.cat([query_feature, time_emb], 0) # (S+1) B D
        
        # add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature)
        if self.img_pos_emb:
            img_feature = self.sequence_pos_encoder(img_feature)
        
        retrieved_feature = self.sequence_pos_encoder(retrieval_feature)
        
        query_feature_wo_guide = torch.cat([query_feature, img_feature]) # (2S + 1) B D
        query_feature_with_guide = torch.cat([query_feature, retrieval_feature])
        
        # transformer process
        motion_feature_wo_guide = self.seqTransEncoder(query_feature_wo_guide) # (2S + 1) B D
        motion_feature_with_guide = self.seqTransEncoder(query_feature_with_guide)
        
        # decode
        motion_feature_wo_guide = rearrange(motion_feature_wo_guide[:nframes], "S B D -> B S D") # (2S + 1) B D -> B S D
        motion_feature_with_guide = rearrange(motion_feature_with_guide[:nframes], "S B D -> B S D") # (2S + 1) B D -> B S D
        
        pred_noise_dict = {}
        for key in self.output_module_dict.keys():
            pred_noise_wo_guide = query[key] + self.output_module_dict[key](motion_feature_wo_guide)
            pred_noise_with_guide = query[key] + self.output_module_dict[key](motion_feature_with_guide)
            pred_noise_dict[key] = guidance_rate*pred_noise_with_guide - ((1 - guidance_rate) * pred_noise_wo_guide)
        
        return pred_noise_dict

class Latent_Transformer_Predictor(torch.nn.Module):
    def __init__(self, ae_latent_dim, transformer_latent_dim, 
                    ff_size=1024, num_layers=8, num_heads=4, dropout=0.0,
                    activation="gelu", img_guidance_rate=1.0, **kargs):
        super().__init__()
        """
        This model is based of Motion Diffusion Model from https://arxiv.org/pdf/2209.14916.pdf
        The image features is fed into the model insted of the text features.
        """
        # configuration
        self.latent_dim = transformer_latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.img_guidance_rate = img_guidance_rate
        
        # query_embedding
        self.query_emb_model = torch.nn.Sequential(
                            LinearBlock(ae_latent_dim, ae_latent_dim, activation=activation),
                            LinearBlock(ae_latent_dim, transformer_latent_dim, activation=activation))
        
        # embedding noise step
        self.steps_encoder = StepEncoding(self.latent_dim)
        
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
        self.output_module = torch.nn.Sequential(
                            LinearBlock(self.latent_dim, int(self.latent_dim / 2), activation=activation),
                            LinearBlock(int(self.latent_dim / 2), int(self.latent_dim / 2), activation=activation),
                            LinearBlock(int(self.latent_dim / 2), ae_latent_dim, activation="none"))

    def forward(self, img_feature, query, time_step):
        # embedding the pose to the vector
        query_feature = torch.unsqueeze(self.query_emb_model(query["latent"]), 1) # B 1 D
        query_feature = rearrange(query_feature, "B S D -> S B D") # 1 B D
        
        # add time embedding 
        time_emb = self.steps_encoder(time_step)
        query_feature = torch.cat([query_feature, time_emb], 0) # (1+1) B D
        
        # add positional encoding
        img_feature = self.sequence_pos_encoder(img_feature)
        
        if random.random() <= self.img_guidance_rate:
            cat_feature = torch.cat([query_feature, img_feature]) # (S + 2) B D
        else:
            cat_feature = query_feature
        
        # transformer process
        motion_feature = self.seqTransEncoder(cat_feature) # (S + 2) B D
        
        # decode
        motion_feature = motion_feature[0] # (S + 2) B D -> B D
        
        pred_noise_dict = {}
        pred_noise_dict["latent"] = query["latent"] + self.output_module(motion_feature)
        return pred_noise_dict

class Image_feature_extractor_model(torch.nn.Module):
    def __init__(self, extractor_name, img_feature_dim, down_scale, num_vec=8):
        super().__init__()
        
        if "query_uv_feature" == extractor_name:
            self.extractor = query_uv_extractor(down_scale)
        else:
            raise ValueError(f"Invalid key: {extractor_name} is invalid key for the Image-feature_extractor_model (in feature_extractory.py)")
        
    def forward(self, x, uv):
        extractor_dict, extractor_info = self.extractor(x, uv)
        return extractor_dict, extractor_info

class query_uv_extractor(torch.nn.Module):
    
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
        y: pose B,S,2

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
        
        y = torch.unsqueeze(y, 1)
        feature = grid_sample(x, y) # B, C, N, S
        # feature = grid_sample(x, y, mode='bilinear', padding_mode='zeros', align_corners=True) # B, C, N, S
        feature = rearrange(feature, 'B C N S -> S N B C') # B, N, S, C
        feature = torch.squeeze(feature, 1)
        output_dict["img_feature"] = feature
        return output_dict, debug_info

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

###########################################
####### comparisons #######################
###########################################

class SPE_Energy_Predictor(torch.nn.Module):
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
        self.ife = Image_feature_extractor_model(extractor_name, emb_dim, 1)
        
        # predictor
        self.predictor = Transformer_Energy_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)

    def forward(self, img, query, time_step, with_feature=False):
        device = img.device
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        uv = query['uv'].to(device)
        img_feature_dict, _ = self.ife(img_feature, uv)
        
        output_energy = self.predictor(img_feature_dict["img_feature"], query, time_step)
            
        return output_energy
        
    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        self.img_feature = img_feature
        return img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        device = img.device
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        uv = query['uv'].to(device)
        img_feature_dict, _ = self.ife(img_feature, uv) # TODO: dict?
        
        return img_feature_dict["img_feature"]

class Transformer_Energy_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim, 
                    ff_size=1024, num_layers=8, num_heads=4, dropout=0.0,
                    activation="gelu", img_guidance_rate=1.0, **kargs):
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
        
        # embedding pose to vector
        self.query_emb_model = Query_emb_model(query_keys, query_dims, latent_dim)
        
        # embedding noise step
        self.steps_encoder = StepEncoding(latent_dim)
        
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
        self.output_module = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), 1, activation="none"))

    def forward(self, img_feature, query, time_step):
        # embedding the pose to the vector
        query_feature = self.query_emb_model(query)
        bs, nframes, nfeats = query_feature.shape
        query_feature = rearrange(query_feature, "B S D -> S B D")
        
        # add time embedding 
        time_emb = self.steps_encoder(time_step)
        query_feature = torch.cat([query_feature, time_emb], 0) # (S+1) B D
        
        # add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature)
        img_feature = self.sequence_pos_encoder(img_feature)
        
        if random.random() <= self.img_guidance_rate:
            cat_feature = torch.cat([query_feature, img_feature]) # (2S + 1) B D
        else:
            cat_feature = query_feature
        
        # transformer process
        motion_feature = self.seqTransEncoder(cat_feature) # (2S + 1) B D
        
        # decode
        motion_feature = rearrange(motion_feature[:nframes], "S B D -> B S D") # (2S + 1) B D -> B S D
        
        pred_energy = torch.sum(self.output_module(motion_feature), 1, keepdim=True) # B, S, D -> B

        return pred_energy