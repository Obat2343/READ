import math
import torch
import statistics

from tqdm import tqdm
from einops import rearrange, repeat

from ..dataset import RLBench_DMOEBM
from ..retrieval import Direct_Retrieval

class Forward_diffusion():
    
    def __init__(self, max_timesteps, start=1e-5, end=2e-2):
        self.max_timesteps = max_timesteps
        self.start_beta = start
        self.end_beta = end
        
        self.betas = torch.abs(torch.cat([torch.tensor([0.]), torch.linspace(start, end, max_timesteps)],0))
        alphas = 1. - self.betas
        alphas = torch.clip(alphas, 1e-8, 1)
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def get_values_from_timestep_for_sampling(self, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        beta_t = self.get_index_from_list(self.betas, t, x_shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_shape)
        return beta_t, sqrt_one_minus_alphas_cumprod_t, sqrt_recip_alphas_t, posterior_variance_t
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise_action_dict = {}
        shifted_action_dict = {}
        noise_dict = {}
        for key in action_dict.keys():
            noise = torch.randn_like(action_dict[key])
            
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            shifted_action_dict[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            noise_action_dict[key] = shifted_action_dict[key] + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
            noise_dict[key] = noise.to(device)
        
        return noise_action_dict, noise_dict, shifted_action_dict

    def get_mean_and_std(self, action_dict, t, device="cpu"):
        std = {}
        mean = {}
        for key in action_dict.keys():
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            mean[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            std[key] = sqrt_one_minus_alphas_cumprod_t.to(device)
        
        return mean, std
    
    def get_step(self, diff):
        alpha_values = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        step = torch.where(alpha_values > diff)[0][0]
        return step

class Improved_Forward_diffusion():
    
    def __init__(self, max_timesteps, s=0.008, bias=0.0):
        self.max_timesteps = max_timesteps
        self.s = s
        t_div_T = torch.arange(0, max_timesteps+1) / max_timesteps
        f = torch.pow(torch.cos((t_div_T + self.s) / (1 + self.s) * (math.pi / 2)), 2)
        alphas_cumprod = bias + ((1 - bias) * (f / f[0]))
        
        temp = alphas_cumprod[:-1]
        temp = torch.cat([torch.tensor([1.]), temp],0 )
        self.betas = torch.clip(1 - (alphas_cumprod / temp), max=0.9999)

        alphas = 1. - self.betas
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def get_values_from_timestep_for_sampling(self, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        beta_t = self.get_index_from_list(self.betas, t, x_shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_shape)
        return beta_t, sqrt_one_minus_alphas_cumprod_t, sqrt_recip_alphas_t, posterior_variance_t
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise_action_dict = {}
        shifted_action_dict = {}
        noise_dict = {}
        for key in action_dict.keys():
            noise = torch.randn_like(action_dict[key])
            
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            shifted_action_dict[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            noise_action_dict[key] = shifted_action_dict[key] + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
            noise_dict[key] = noise.to(device)
        
        return noise_action_dict, noise_dict, shifted_action_dict

    def get_mean_and_std(self, action_dict, t, device="cpu"):
        std = {}
        mean = {}
        for key in action_dict.keys():
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            mean[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            std[key] = sqrt_one_minus_alphas_cumprod_t.to(device)
        
        return mean, std
    
    def get_step(self, diff):
        alpha_values = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        step = torch.where(alpha_values > diff)[0][0]
        return step

class Forward_DSM():
    
    def __init__(self, max_timesteps=100, start=1e-2, end=2e-1):
        self.max_timesteps = max_timesteps
        self.start_beta = start
        self.end_beta = end
        
        self.stds = torch.cat([torch.tensor([0.]), torch.linspace(start, end, max_timesteps)],0)
        self.sigmas = torch.pow(self.stds, 2)
        
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise_action_dict = {}
        scaled_noise_dict = {}
        stds_vec = self.get_index_from_list(self.stds, t, t.shape)
        for key in action_dict.keys():
            noise = torch.randn_like(action_dict[key]).to(device)
            stds = self.get_index_from_list(self.stds, t, action_dict[key].shape)
            sigmas = torch.pow(stds, 2)
            
            scaled_noise = stds.to(device) * noise
            noise_action_dict[key] = action_dict[key] + scaled_noise
            scaled_noise_dict[key] = scaled_noise
        
        return noise_action_dict, scaled_noise_dict, stds_vec

class Forward_Latent_DSM():
    
    def __init__(self, ae, max_timesteps=100, start=1e-2, end=2e-1):
        self.max_timesteps = max_timesteps
        self.start_beta = start
        self.end_beta = end
        
        self.stds = torch.cat([torch.tensor([0.]), torch.linspace(start, end, max_timesteps)],0)
        self.sigmas = torch.pow(self.stds, 2)
        self.ae = ae
        
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noised_action_dict = {}
        noise_dict = {}
        stds_vec = self.get_index_from_list(self.stds, t, t.shape)
        
        with torch.no_grad():
            z = self.ae.encode(action_dict)
            
            noise = torch.rand_like(z).to(device)
            stds = self.get_index_from_list(self.stds, t, z.shape)
            scaled_latent_noise = stds.to(device) * noise
        
            noised_z = z + scaled_latent_noise
            noised_action_dict = self.ae.decode(noised_z)
            
        for key in noised_action_dict.keys():
            noise_dict[key] = noised_action_dict[key] - action_dict[key]
        
        return noised_action_dict, noise_dict, stds_vec

def calculate_end(cfg, rank, max_step, start=1e-5, iteration=5000, rot_mode="6d", target_mode="max", sigma=1.):
    print("##############################")
    print("Calculate hyper-parameter for retrieval diffusion")
    print("loading data")
    dataset = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=100, rot_mode=rot_mode, img_aug=False)
    dataset.without_image = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)

    print("setting retrieval")
    Retriever = Direct_Retrieval(dataset)   
    
    print("start getting kNN")
    for data in dataloader:
        _, query = data
        near_queries, _, _ = Retriever.retrieve_k_sample(query, k=cfg.DIFFUSION.MAX_RANK+1)
    print("end\n")

    print(f"task: {cfg.DATASET.RLBENCH.TASK_NAME} rank: {rank}")
    diff_dict = {}
    retrieved_query = {}
    for key in near_queries.keys():
        retrieved_query[key] = near_queries[key][:,rank]
        diff_dict[key] = torch.pow(retrieved_query[key] - query[key], 2)

    uv_diff_list = []
    z_diff_list = []
    rotation_diff_list = []
    grasp_diff_list = []

    for i in range(1000):
        uv_diff_list.append(torch.max(diff_dict["uv"][i]).item())
        z_diff_list.append(torch.max(diff_dict["z"][i]).item())
        rotation_diff_list.append(torch.max(diff_dict["rotation"][i]).item())
        grasp_diff_list.append(torch.max(diff_dict["grasp_state"][i]).item())

    if target_mode == "max":
        target = max([max(uv_diff_list), max(z_diff_list)])
    elif target_mode == "mean":
        target_list = uv_diff_list + z_diff_list
        target = sum(target_list) / len(target_list)
    elif target_mode == "median":
        target_list = uv_diff_list + z_diff_list
        target = statistics.median(target_list)
    elif "std" in target_mode:
        target_list = uv_diff_list + z_diff_list
        
        mean = sum(target_list) / len(target_list)
        std_dev = statistics.stdev(target_list)
        
        if target_mode == "std1":
            target = mean + (2 * std_dev)
        elif target_mode == "std2":
            target = mean + (2 * std_dev)
        else:
            raise NotImplementedError("TODO")
            
    print(f"diff_{target}")

    x = torch.nn.parameter.Parameter(torch.tensor(2e-5, requires_grad=True))
    optimizer = torch.optim.Adam([x], lr=1e-8)
    target = (sigma**2)/((sigma**2) + target)

    print("start optimizing hyper-params")
    for i in tqdm(range(iteration)):
        value = 1
        for t in range(0, max_step):
            value = value * (1 - start - x*t) 

        loss = torch.abs(value - target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("done")

    with torch.no_grad():
        result = start + x*max_step

    print(f"target: {target} loss: {loss.item()} result: (start, end) = ({start}, {result}) \n")
    return result.item()

def calculate_end_for_cosine(cfg, rank, rot_mode="6d"):
    print("Calculate hyper-parameter for retrieval diffusion")
    print("loading data")
    dataset = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=100, rot_mode=rot_mode, img_aug=False)
    dataset.without_image = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)

    print("setting retrieval")
    Retriever = Direct_Retrieval(dataset)   
    
    print("start getting kNN")
    for data in dataloader:
        _, query = data
        near_queries, _, _ = Retriever.retrieve_k_sample(query, k=cfg.DIFFUSION.MAX_RANK+1)
    print("end\n")

    print(f"task: {cfg.DATASET.RLBENCH.TASK_NAME} rank: {rank}")
    diff_dict = {}
    retrieved_query = {}
    for key in near_queries.keys():
        retrieved_query[key] = near_queries[key][:,rank]
        diff_dict[key] = torch.pow(retrieved_query[key] - query[key], 2)

    uv_diff_list = []
    z_diff_list = []
    rotation_diff_list = []
    grasp_diff_list = []

    for i in range(1000):
        uv_diff_list.append(torch.max(diff_dict["uv"][i]))
        z_diff_list.append(torch.max(diff_dict["z"][i]))
        rotation_diff_list.append(torch.max(diff_dict["rotation"][i]))
        grasp_diff_list.append(torch.max(diff_dict["grasp_state"][i]))

    target = max([max(uv_diff_list), max(z_diff_list)])
    target = 1/(1 + target)

    print(f"target: {target} \n")
    return target.item()

class SAMR_Loss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
    
    def forward(self, pred_noise, gt_noise, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_noise.keys():
            loss = self.l1(pred_noise[key], gt_noise[key])
            loss_dict[f"{mode}/{key}"] = loss.item()
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict

class Noise_Sampler():
    
    def __init__(self, cfg, max_noise_steps=20, vae="none"):
        self.max_steps = max_noise_steps
        
        # define noise scale
        if cfg.SAMR.NOISE.SCALE == "custom":
            self.min_std = cfg.SAMR.NOISE.CUSTOM.MIN
            self.max_std = cfg.SAMR.NOISE.CUSTOM.MAX
        elif cfg.SAMR.NOISE.SCALE == "auto":
            raise ValueError("TODO")
        else:
            raise ValueError("Invalid mode. cfg.NOISE.SCALE should be custom or auto")
        
        # get noise class
        self.noise_type = cfg.SAMR.NOISE.TYPE
        if self.noise_type == "gaussian":
            self.noise_sampler = Gaussian_Noise(max_noise_steps=self.max_steps, min_noise_std=self.min_std, max_noise_std=self.max_std)
        elif self.noise_type == "latent-gaussian":
            if vae == "none":
                raise ValueError("Please input pre-trained vae")
            self.noise_sampler = Latent_Gaussian_Noise(vae, max_noise_steps=self.max_steps, min_noise_std=self.min_std, max_noise_std=self.max_std)
            
    def get_noise(self, inputs):
        return self.noise_sampler.get_noise(inputs)

    def get_noised_query(self, query, time_step):
        return self.noise_sampler.get_noised_query(query, time_step)

class Gaussian_Noise():
    def __init__(self, max_noise_steps=20, min_noise_std=0.1, max_noise_std=2.0):
        self.max_noise_steps = max_noise_steps
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
        steps = torch.arange(0, max_noise_steps)
        steps = steps / (max_noise_steps - 1) # normalize
        steps = steps * (max_noise_std - min_noise_std)
        self.noise_std_list = steps + min_noise_std
        
    def get_noise(self, query):
        B, S, _ = query["uv"].shape
        device = query["uv"].device
        random_steps = torch.randint(0, self.max_noise_steps, (B,)).long()
        sampled_std = self.noise_std_list.gather(-1, random_steps).to(device)
        
        sampled_noise = {}
        noised_query = {}
        for key in query.keys():
            if key == "time":
                continue
                
            _, _, D = query[key].shape
            sampled_noise[key] = repeat(sampled_std, "B -> B S D",S=S,D=D) * torch.randn(B, S, D).to(device)
            noised_query[key] = query[key] + sampled_noise[key]
            
        return noised_query, sampled_noise, torch.unsqueeze(sampled_std, 1), random_steps.to(device)
    
    def get_noised_query(self, query, time_step):
        B, S, _ = query["uv"].shape
        device = query["uv"].device
        time_step = torch.ones(B).long() * time_step
        sampled_std = self.noise_std_list.gather(-1, time_step).to(device)
        
        sampled_noise = {}
        noised_query = {}
        for key in query.keys():
            if key == "time":
                continue
                
            _, _, D = query[key].shape
            sampled_noise[key] = repeat(sampled_std, "B -> B S D",S=S,D=D) * torch.randn(B, S, D).to(device)
            noised_query[key] = query[key] + sampled_noise[key]
            
        return noised_query
    
class Latent_Gaussian_Noise():
    def __init__(self, vae, max_noise_steps=20, min_noise_std=0.1, max_noise_std=2.0):
        self.max_noise_steps = max_noise_steps
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
        steps = torch.arange(0, max_noise_steps)
        steps = steps / (max_noise_steps - 1) # normalize
        steps = steps * (max_noise_std - min_noise_std)
        self.noise_std_list = steps + min_noise_std
        
        self.vae = vae

    def get_noise(self, query):
        B, _, _ = query["uv"].shape
        device = query["uv"].device
        self.vae = self.vae.to(device)

        with torch.no_grad():
            z = self.vae.encode(query)
            _, D = z.shape

        random_steps = torch.randint(0, self.max_noise_steps, (B,)).long()
        sampled_std = self.noise_std_list.gather(-1, random_steps).to(device)
        sampled_z_noise = repeat(sampled_std, "B -> B D", D=D) * torch.randn(B, D).to(device)
        noised_z = z + sampled_z_noise

        with torch.no_grad():
            noised_query = self.vae.decode(noised_z)

        sampled_noise = {}
        for key in noised_query.keys():
            sampled_noise[key] = noised_query[key] - query[key]

        return noised_query, sampled_noise, torch.unsqueeze(sampled_std, 1), random_steps.to(device)
    
    def get_noised_query(self, query, time_step):
        B, _, _ = query["uv"].shape
        device = query["uv"].device
        self.vae = self.vae.to(device)

        with torch.no_grad():
            z = self.vae.encode(query)
            _, D = z.shape

        time_step = torch.ones(B).long() * time_step
        sampled_std = self.noise_std_list.gather(-1, time_step).to(device)
        sampled_z_noise = repeat(sampled_std, "B -> B D", D=D) * torch.randn(B, D).to(device)
        noised_z = z + sampled_z_noise

        with torch.no_grad():
            noised_query = self.vae.decode(noised_z)

        sampled_noise = {}
        for key in noised_query.keys():
            sampled_noise[key] = noised_query[key] - query[key]

        return noised_query