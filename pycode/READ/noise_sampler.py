import torch
from einops import rearrange, repeat

def Normal_Noise():

    def sampler(query):
        """
        Args
            query (dict): motion query or latent query. This dict should include tensors.
        """
        noise_dict = {}
        for key in query.keys():
            noise_dict[key] = torch.randn_like(query[key])
            noise_dict[key] = noise_dict[key].to(query[key].device)
        return noise_dict
    
    return sampler