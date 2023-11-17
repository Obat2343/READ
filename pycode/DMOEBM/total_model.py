
import torch

from ..model.resnet_module import Resnet_Like_Decoder, Resnet_Like_Encoder
from .predictor import Predictor
from .obs_encoder import obs_emb_model

class DMOEBM(torch.nn.Module):
    def __init__(self, query_list, query_dims, img_size=256, input_dim=4, dims=[96, 192, 384, 768], enc_depths=[3,3,9,3], enc_layers=['conv','conv','conv','conv'],
                 dec_depths=[3,3,3], dec_layers=['conv','conv','conv'], enc_act="gelu", enc_norm="layer", dec_act="gelu", dec_norm="layer", drop_path_rate=0.1,
                 extractor_name="query_uv_feature", predictor_name="HIBC_Transformer_with_cat_feature", num_attn_block=4,
                 mlp_drop=0.1, query_emb_dim=128):
        """
        Args:
        img_size (int): Size of image. We assume image is square.
        input_dim (int): Size of channel. 3 for RGB image.
        enc_depths (list[int]): Number of blocks at each stage for encoder.
        dec_depths (list[int]): Number of blocks at each stage for decoder.
        predictor_depth (int): Number of blocks at predicotr. This value is for IABC predictor.
        dims (list[int]): The channel size of each feature map.
        enc_layers (list[str]): Name of layer at each stage of encoder.
        dec_layers (list[str]): Name of layer at each stage of decoder.
        predictor (str): Name of predictor
        predictor_prob_func (str): Function to change the vector to probability. This function is for IABC.
        act (str): Activation function.
        norm (str): Normalization function.
        atten (str): Name of attention. If layer name is atten, indicated atten layer is used.
        drop_path (float): Stochastic depth rate for encoder and decoder. Default: 0.1
        """
        super().__init__()
        self.img_size = img_size
        if query_emb_dim == 0:
            emb_dim = dims[0] // 2
        else:
            emb_dim = query_emb_dim

        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        self.predictor = Predictor(extractor_name, predictor_name, emb_dim, query_list, query_dims, drop=mlp_drop, img_emb_dim=emb_dim, num_attn_block=num_attn_block)
    
    def forward(self, img, query, with_feature=False):
        debug_info = {}
        if with_feature == False:
            img_feature = self.enc(img)
            img_feature = self.dec(img_feature)
            self.img_feature = img_feature
        else:
            img_feature = self.img_feature

        output_dict, pred_info = self.predictor(img_feature, query)
        for key in pred_info.keys():
            debug_info[key] = pred_info[key]
        return output_dict, debug_info
