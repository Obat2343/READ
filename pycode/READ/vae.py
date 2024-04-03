import torch
import numpy as np

from einops import rearrange, repeat

from ..model.base_module import LinearBlock

########### Model ###########

class Single_Class_TransformerVAE(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_frames, num_classes=1,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0., activation="gelu", **kargs):
        super().__init__()
        
        self.encoder = Encoder_TRANSFORMER(query_keys, query_dims, num_classes=num_classes,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        self.decoder = Decoder_TRANSFORMER(query_keys, query_dims, num_frames, num_classes=num_classes,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        self.latent_dim = latent_dim

    def forward(self, query):
        mu, logvar = self.encoder(query)
        z = self.reparameterize(mu, logvar)
        pred_dict = self.decoder(z)

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

        for key in pred_dict.keys():
            pred_dict[key] = rearrange(pred_dict[key], "(B N) S D -> B N S D", B=B)

        return pred_dict, z
    
    def reconstruct(self, x):
        mu, log_var = self.encoder(x)
        pred_dict = self.decoder(mu)
        return pred_dict, mu

    def encode(self, x):
        mu, _ = self.encoder(x)
        return {"latent": mu}
    
    def decode(self, z):
        pred_dict = self.decoder(z["latent"])
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
                            LinearBlock(int(latent_dim / 2), dim, activation="none"))
        
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

class Multi_Latent_TransformerVAE(torch.nn.Module):
    """
    Based on https://arxiv.org/pdf/2212.04048.pdf
    """

    def __init__(self, query_keys, query_dims, num_frames, num_latent=8,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0., activation="gelu", **kargs):
        super().__init__()
        
        self.encoder = Multi_Latent_Encoder_TRANSFORMER(query_keys, query_dims, num_latent=num_latent,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        self.decoder = Multi_Latent_Decoder_TRANSFORMER(query_keys, query_dims, num_frames, num_latent=num_latent,
                            latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                            num_heads=num_heads, dropout=dropout, activation=activation)
        
        self.latent_dim = latent_dim

    def forward(self, query):
        mu, logvar = self.encoder(query)
        z = self.reparameterize(mu, logvar)
        pred_dict = self.decoder(z)

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
        z = torch.randn(num_samples, self.num_latent, self.latent_dim)

        z = z.to(device)
        pred_dict = self.decoder(z)
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

        std = repeat(std, "B M D -> B N M D", N=sample_num)
        mu = repeat(mu, 'B M D -> B N M D', N=sample_num)
        eps = torch.randn_like(std)
        z = eps * std * noise_std + mu
        z = rearrange(z, "B N M D -> (B N) M D")

        pred_dict = self.decoder(z)

        for key in pred_dict.keys():
            pred_dict[key] = rearrange(pred_dict[key], "(B N) S D -> B N S D", B=B)

        return pred_dict, z
    
    def reconstruct(self, x):
        mu, log_var = self.encoder(x)
        pred_dict = self.decoder(mu)
        return pred_dict, mu

    def encode(self, x):
        mu, _ = self.encoder(x)
        return {"latent": mu}
    
    def decode(self, z):
        pred_dict = self.decoder(z["latent"])
        return pred_dict

class Multi_Latent_Encoder_TRANSFORMER(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_latent=8,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                    activation="gelu", **kargs):
        super().__init__()
        
        self.query_keys = query_keys
        self.query_dims = query_dims
        self.num_latent = num_latent
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        self.muQuery = torch.nn.Parameter(torch.randn(self.num_latent, self.latent_dim))
        self.sigmaQuery = torch.nn.Parameter(torch.randn(self.num_latent, self.latent_dim))
        
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
        muquery = repeat(self.muQuery, "M D -> M B D", B=bs)
        sigmaquery = repeat(self.sigmaQuery, "M D -> M B D", B=bs)
        xseq = torch.cat((muquery, sigmaquery, emb_vec), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        final = self.seqTransEncoder(xseq)
        mu = final[:self.num_latent]
        logvar = final[self.num_latent:2*self.num_latent]
            
        return mu, logvar

class Multi_Latent_Decoder_TRANSFORMER(torch.nn.Module):
    def __init__(self, query_keys, query_dims, num_frames, num_latent=8,
                    latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                    ablation=None, **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.num_latent = num_latent
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
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
                            LinearBlock(int(latent_dim / 2), dim, activation="none"))
        
        self.output_module_dict = torch.nn.ModuleDict(module_dict)
        
    def forward(self, z):
        latent_num, bs, latent_dim = z.shape
        nframes = self.num_frames
        
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z)
        
#         output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        output = rearrange(output, "S B D -> B S D")
        
        pred_dict = {}
        for key in self.output_module_dict.keys():
            pred_dict[key] = self.output_module_dict[key](output)
        
        return pred_dict

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

    def forward(self, queries):
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
        keys = list(queries.keys())
        keys.sort()

        query_list = []
        for key in keys:
            if key in self.register_query_keys:
                query_list.append(queries[key])
        
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

############# Loss ###############

class VAE_Loss(torch.nn.Module):
    
    def __init__(self, rot_mode="6d", kld_weight=0.01, device="cuda"):
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
