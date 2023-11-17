import torch
import torch.nn as nn

class obs_emb_model(torch.nn.Module):

    def __init__(self, obs_dim, emb_dim, drop=0., act="gelu"):
        super().__init__()
        self.linear = self.make_linear_model(obs_dim, emb_dim, act, drop)

    def forward(self, obs):
        emb_obs = self.linear(obs)
        return emb_obs
        
    def make_linear_model(self, input_dim, output_dim, act, drop):
        model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            nn.Dropout(drop),
            nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            torch.nn.Dropout(drop),
            nn.Linear(output_dim * 2, output_dim))
        return model

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