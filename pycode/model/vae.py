import torch
import numpy as np

from einops import rearrange, repeat
from typing import List

from .Regression_loss import Motion_Loss
from ..model.base_module import LinearBlock
from ..misc import get_pos, load_checkpoint

class Input_Converter():
    
    def __init__(self, query_key=["uv","z","rotation","grasp_state"], query_dims=[2,1,6,1], frame=101):
        
        # sort query key and dim
        temp = sorted(zip(query_key, query_dims), key=lambda x: x[0])
        query_key, query_dims = map(list, zip(*temp))

        # calculate each vectorized query dim
        dim_value = 0
        query_dim_list = []
        for dim in query_dims:
            query_dim_list.append([dim_value, dim_value + dim*frame])
            dim_value += dim*frame

        self.query_key =  query_key
        self.query_dim_list = query_dim_list
        self.assert1 = True
        self.assert2 = True
        self.frame = frame

    def query2vec(self, query):
        vec_input = []
        query_dim_list = []
        dim_value = 0
        for i, key in enumerate(self.query_key):
            B, S, D = query[key].shape
            vec_input.append(rearrange(query[key], "B S D -> B (S D)"))
            query_dim_list.append([dim_value, dim_value + S*D])
            dim_value += S * D

            if query_dim_list[i] != self.query_dim_list[i]:
                print("query_dim_list")
                print(query_dim_list[i])
                print("self.query_dim_list")
                print(self.query_dim_list[i])
                raise ValueError("Invalid shape")

        vec_input = torch.cat(vec_input, 1)
        return vec_input
    
    def vec2query(self, vec, intrinsic="none", time="none"):
        query = {}
        for key, dim_range in zip(self.query_key, self.query_dim_list):
            query_ins = vec[:,dim_range[0]:dim_range[1]]
            query_ins = rearrange(query_ins, "B (S D) -> B S D", S=self.frame)
            query[key] = query_ins
        
        if type(time) != str:
            query["time"] = time
        elif self.assert1:
            print("============================")
            print("vec2query returns query which does not include time information")
            self.assert1 = False
        
        if type(intrinsic) != str:
            query = get_pos(query, intrinsic)
        elif self.assert2:
            print("============================")
            print("vec2query returns query which does not include pos information")
            self.assert2 = False
            
        return query

class Norm_and_Denorm():
    
    def __init__(self, nf_input):
        self.mean, self.std = torch.std_mean(nf_input, dim=0)
        
    def norm(self, nf_input):
        device = nf_input.device
        return (nf_input / self.std.to(device)) + self.mean.to(device)
    
    def denorm(self, nf_input):
        device = nf_input.device
        return (nf_input - self.mean.to(device)) * self.std.to(device)

###########
### VAE ###
###########

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


# only for ablation / not used in the final model
class TimeEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

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

    def forward(self, query):
        emb_vec = self.query_emb_model(query)
        bs, nframes, nfeats = emb_vec.shape
        emb_vec = rearrange(emb_vec, "B S D -> S B D")

        # adding the mu and sigma queries
        index = [0] * bs
        xseq = torch.cat((self.muQuery[index][None], self.sigmaQuery[index][None], emb_vec), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        final = self.seqTransEncoder(xseq)
        mu = final[0]
        logvar = final[1]
            
        return mu, logvar
    

class Decoder_TRANSFORMER(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_frames, num_classes=1,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                    ablation=None, **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.actionBiases = torch.nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = torch.nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=activation)
        self.seqTransDecoder = torch.nn.TransformerDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers)
        
        module_dict = {}
        for key, dim in zip(query_keys, query_dims):
            if key == "time":
                continue
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2)),
                            LinearBlock(int(latent_dim / 2), dim))
        
        self.output_module_dict = torch.nn.ModuleDict(module_dict)
        
    def forward(self, z):
        bs, latent_dim = z.shape
        nframes = self.num_frames
        
        # shift the latent noise vector to be the action noise
        z = z + self.actionBiases[0]
        z = z[None]  # sequence of size 1
            
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z)
        
#         output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        output = rearrange(output, "S B D -> B S D")
        
        pred_dict = {}
        for key in self.output_module_dict.keys():
            pred_dict[key] = self.output_module_dict[key](output)
        
        return pred_dict

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
                query_list.append(querys[key])
        
        query_cat = torch.cat(query_list, 2)
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

class Single_Class_TransformerVAE(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_frames, num_classes=1,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0., activation="gelu",
                    intrinsic="none", **kargs):
        super().__init__()
        
        self.encoder = Encoder_TRANSFORMER(query_keys, query_dims, num_classes=num_classes,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        self.decoder = Decoder_TRANSFORMER(query_keys, query_dims, num_frames, num_classes=num_classes,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        self.intrinsic = intrinsic
        self.latent_dim = latent_dim

    def forward(self, query):
        mu, logvar = self.encoder(query)
        z = self.reparameterize(mu, logvar)
        pred_dict = self.decoder(z)

        if type(self.intrinsic) != str:
            pred_dict = get_pos(pred_dict, self.intrinsic)

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

    def sample(self,
                num_samples:int,
                device:str):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)
        pred_dict = self.decoder(z)

        if type(self.intrinsic) != str:
            pred_dict = get_pos(pred_dict, self.intrinsic)

        return pred_dict

    def sample_from_query(self, query, sample_num, noise_std=1.):
        """
        : param x: (torch.tensor) :: shape -> (batch_size, dim)
        : param sample_num: (int)
        : param nois_level: (float) :: noise is sampled from the normal distribution. noise std is multiplied to predicted std. 
        """
        mu, log_var = self.encoder(query)
        B = mu.shape[0]
        std = torch.exp(0.5 * log_var)

        std = repeat(std, "B D -> B N D", N=sample_num)
        mu = repeat(mu, 'B D -> B N D', N=sample_num)
        eps = torch.randn_like(std)
        z = eps * std * noise_std + mu
        z = rearrange(z, "B N D -> (B N) D")

        pred_dict = self.decoder(z)

        if type(self.intrinsic) != str:
            pred_dict = get_pos(pred_dict, self.intrinsic)

        for key in pred_dict.keys():
            pred_dict[key] = rearrange(pred_dict[key], "(B N) S D -> B N S D", B=B)

        return pred_dict, z
    
    def reconstruct(self, x):
        mu, log_var = self.encoder(x)
        
        pred_dict = self.decoder(mu)
        if type(self.intrinsic) != str:
            pred_dict = get_pos(pred_dict, self.intrinsic)

        return pred_dict, mu

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        pred_dict = self.decoder(z)
        if type(self.intrinsic) != str:
            pred_dict = get_pos(pred_dict, self.intrinsic)

        return pred_dict


class VAE(torch.nn.Module):
    
    def __init__(self, 
                    input_dim: int,
                    latent_dim: int,
                    query_key: List = ["uv","z","rotation","grasp_state"],
                    query_dims: List = [2, 1, 6, 1],
                    hidden_dims: List = [512, 256, 128, 64, 32],
                    intrinsic = "none"):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        self.encoder = torch.nn.Sequential(*self.make_encoder_layer())
        self.decoder = torch.nn.Sequential(*self.make_decoder_layer())
        self.fc_mu = LinearBlock(self.hidden_dims[-1], self.latent_dim, activation="none", norm="none")
        self.fc_var = LinearBlock(self.hidden_dims[-1], self.latent_dim, activation="none", norm="none")
        self.converter = Input_Converter(query_key, query_dims=query_dims)

        self.intrinsic = intrinsic
        
    def make_encoder_layer(self, act="relu", norm="none"):
        layer_list = [LinearBlock(self.input_dim, self.hidden_dims[0])]
        
        for index in range(len(self.hidden_dims) - 1):
            layer_list.append(LinearBlock(self.hidden_dims[index], self.hidden_dims[index + 1], activation=act, norm=norm))
            
        return layer_list
    
    def make_decoder_layer(self, act="relu", norm="none"):
        layer_list = [LinearBlock(self.latent_dim, self.hidden_dims[-1], activation=act, norm=norm)]
        
        for index in range(len(self.hidden_dims) - 1, 0, -1):
            layer_list.append(LinearBlock(self.hidden_dims[index], self.hidden_dims[index - 1], activation=act, norm=norm))
        
        layer_list.append(LinearBlock(self.hidden_dims[0], self.input_dim, activation="none", norm="none"))
        return layer_list
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, query):
        time = query["time"]
        x = self.converter.query2vec(query)
        latent = self.encoder(x)
        
        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        
        z = self.reparameterize(mu, log_var)
        
        recons = self.decoder(z)
        pred_dict = self.converter.vec2query(recons, intrinsic=self.intrinsic, time=time)
        return pred_dict, z, mu, log_var
    
    def sample(self,
                num_samples:int,
                device:str):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)
        recons = self.decoder(z)
        pred_dict = self.converter.vec2query(recons, intrinsic=self.intrinsic)
        return pred_dict

    def sample_from_query(self, query, sample_num, noise_std=1.):
        """
        : param x: (torch.tensor) :: shape -> (batch_size, dim)
        : param sample_num: (int)
        : param nois_level: (float) :: noise is sampled from the normal distribution. noise std is multiplied to predicted std. 
        """
        x = self.converter.query2vec(query)
        B = x.shape[0]
        latent = self.encoder(x)
        
        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        
        std = torch.exp(0.5 * log_var)

        std = repeat(std, "B D -> B N D", N=sample_num)
        mu = repeat(mu, 'B D -> B N D', N=sample_num)
        eps = torch.randn_like(std)
        z = eps * std * noise_std + mu
        
        recons = self.decoder(z)
        pred_dict = self.converter.vec2query(recons, intrinsic=self.intrinsic)

        for key in pred_dict.keys():
            pred_dict[key] = rearrange(pred_dict[key], "(B N) S D -> B N S D", B=B)

        return pred_dict, z
    
    def reconstruct(self, query):
        time = query["time"]
        x = self.converter.query2vec(query)
        latent = self.encoder(x)
        
        z = self.fc_mu(latent)
        recons = self.decoder(z)
        pred_dict = self.converter.vec2query(recons, intrinsic=self.intrinsic, time=time)
        return pred_dict, z

    def encode(self, query):
        time = query["time"]
        x = self.converter.query2vec(query)
        latent = self.encoder(x)
        z = self.fc_mu(latent)
        return z
    
    def decode(self, z):
        recons = self.decoder(z)
        pred_dict = self.converter.vec2query(recons, intrinsic=self.intrinsic)
        return pred_dict

class VAE_Loss(torch.nn.Module):
    
    def __init__(self, rot_mode="6d", kld_weight=0.01, device="cuda"):
        super().__init__()
        self.motion_loss = Motion_Loss(device=device, mode=rot_mode)
        self.kld_weight = kld_weight
        
    def forward(self, pred_dict, gt_dict, mu, log_var, mode="train"):
        
        # Motion Loss
        loss, loss_dict = self.motion_loss(pred_dict, gt_dict, mode=mode)
        
        # KLD loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        # sum
        loss += self.kld_weight * kld_loss
        
        # register loss value to dict for weights and biases
        loss_dict[f"{mode}/KLD"] = kld_loss.item()
        loss_dict[f"{mode}/loss"] = loss.item()
        
        return loss, loss_dict

class VAE_add_noise():
    
    def __init__(self, model_path, vae_name, intrinsic, rot_mode="6d", frame=100, device="cuda", latent_dim=256):
        
        if rot_mode == "6d":
            rot_dim = 6            
        elif rot_mode == "quat":
            rot_dim = 4
        
        input_size = (2 + 1 + rot_dim + 1) * (frame + 1)

        if vae_name == "VAE":
            model = VAE(input_size, latent_dim=latent_dim, intrinsic=intrinsic).to(device)
        elif vae_name == "Transformer_VAE":
            model = Single_Class_TransformerVAE(["uv","z","rotation","grasp_state"],[2,1,rot_dim,1], frame + 1, latent_dim=latent_dim, intrinsic=intrinsic).to(device)
            
        self.model, _, _, _, _ = load_checkpoint(model, model_path)
        self.model.eval()
        self.device = device
        self.intrinsic = intrinsic

    def __call__(self, h_query, noise_std_mul=1.0):
        for key in h_query.keys():
            h_query[key] = h_query[key].to(self.device)

        with torch.no_grad():
            negative_query, _ = self.model.sample_from_query(h_query, 1, noise_std=noise_std_mul)
            
        negative_query["time"] = torch.unsqueeze(h_query["time"], 1)
        
        return negative_query