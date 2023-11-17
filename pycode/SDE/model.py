import torch

from ..model.base_module import LinearBlock

from pycode.model.base_module import LinearBlock

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