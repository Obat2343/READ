import os
import copy
import random
import torch
import pytorch3d

from tqdm import tqdm
from einops import rearrange, reduce, repeat
from scipy.spatial.transform import Rotation as R

from .Motion_Gen import VAE, Single_Class_TransformerVAE
from ..misc import load_checkpoint, get_pos

def inference(sample_method, query, cfg, image, EBM, DMO="none", vae="none", info_dict="none", intrinsic="none", device="cuda"):
    batch_size = image.shape[0]
    
    for key in query.keys():
        query[key] = repeat(query[key], "N S D -> B N S D", B=batch_size)

    if sample_method == "langevin":
        # config
        lr = cfg.SAMPLING.LANGEVIN.STEP_SIZE
        momentum = cfg.SAMPLING.LANGEVIN.MOMENTUM
        max_iteration = cfg.SAMPLING.LANGEVIN.ITERATION * 2
        decay_step = cfg.SAMPLING.LANGEVIN.DECAY_RATIO
        decay_ratio = cfg.SAMPLING.LANGEVIN.DECAY_STEP
        limit_sample = cfg.SAMPLING.LIMIT_SAMPLE
        # get sample
        sample, pred_dict = langevin(query, image, EBM, intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, verbose=True, sort=True, keep_sample=True, limit_sample=limit_sample)

    elif sample_method == "langevin_vae":
        # check
        if type(vae) == str:
            raise ValueError("please change vae")
        if type(intrinsic) == str:
            raise ValueError("please change intrinsic")

        # config
        lr = cfg.SAMPLING.LANGEVIN_VAE.STEP_SIZE
        momentum = cfg.SAMPLING.LANGEVIN_VAE.MOMENTUM
        max_iteration = cfg.SAMPLING.LANGEVIN_VAE.ITERATION * 2
        decay_step = cfg.SAMPLING.LANGEVIN_VAE.DECAY_RATIO
        decay_ratio = cfg.SAMPLING.LANGEVIN_VAE.DECAY_STEP
        limit_sample = cfg.SAMPLING.LIMIT_SAMPLE

        # get sample
        sample, pred_dict = langevin_vae(query, image, EBM, vae, intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, noise=True, verbose=True, sort=True, keep_half=False, keep_sample=True, limit_sample=limit_sample)

    elif sample_method == "DFO":
        # check
        if type(intrinsic) == str:
            raise ValueError("please change intrinsic")
        if type(info_dict) == str:
            raise ValueError("please change info_dict")

        # config
        max_iteration = cfg.SAMPLING.DFO.ITERATION * 2
        noise_range = cfg.SAMPLING.DFO.RATIO
        decay_step = cfg.SAMPLING.DFO.DECAY_RATIO
        decay_ratio = cfg.SAMPLING.DFO.DECAY_STEP

        # get sample
        sample, pred_dict = DFO(query, image, EBM, info_dict, intrinsic, max_iteration=max_iteration,
                            noise_range=noise_range, device=device, do_norm=True, sort=True, verbose=False)

    elif sample_method == "sgd":
        # config
        lr = cfg.SAMPLING.LANGEVIN.STEP_SIZE
        momentum = cfg.SAMPLING.LANGEVIN.MOMENTUM
        max_iteration = cfg.SAMPLING.LANGEVIN.ITERATION * 2
        decay_step = cfg.SAMPLING.LANGEVIN.DECAY_RATIO
        decay_ratio = cfg.SAMPLING.LANGEVIN.DECAY_STEP
        limit_sample = cfg.SAMPLING.LIMIT_SAMPLE
        # get sample
        sample, pred_dict = langevin(query, image, EBM, intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, noise=False, verbose=True, sort=True, keep_sample=True, limit_sample=limit_sample)

    elif sample_method == "DMO":
        # check
        if type(intrinsic) == str:
            raise ValueError("please change intrinsic")
        if type(DMO) == str:
            raise ValueError("please change DMO")

        # config
        iteration = cfg.SAMPLING.DMO.ITERATION
        threshold = cfg.SAMPLING.DMO.THRESHOLD
        limit_sample = cfg.SAMPLING.LIMIT_SAMPLE

        # get sample
        sample, pred_dict = DMO_optimization(query, image, EBM, DMO, intrinsic, max_iteration=iteration, threshold=threshold, sort=True, verbose=True, keep_sample=False, limit_sample=limit_sample, device=device)

    elif sample_method == "DMO_once":
        # check
        if type(intrinsic) == str:
            raise ValueError("please change intrinsic")
        if type(DMO) == str:
            raise ValueError("please change DMO")

        # config
        iteration = 1
        threshold = cfg.SAMPLING.DMO.THRESHOLD
        limit_sample = cfg.SAMPLING.LIMIT_SAMPLE

        # get sample
        sample, pred_dict = DMO_optimization(query, image, EBM, DMO, intrinsic, max_iteration=iteration, threshold=threshold, sort=True, verbose=True, limit_sample=limit_sample, device=device)

    elif sample_method == "DMO_keep":
        # check
        if type(intrinsic) == str:
            raise ValueError("please change intrinsic")
        if type(DMO) == str:
            raise ValueError("please change DMO")

        # config
        iteration = cfg.SAMPLING.DMO.ITERATION
        threshold = cfg.SAMPLING.DMO.THRESHOLD
        limit_sample = cfg.SAMPLING.LIMIT_SAMPLE

        # get sample
        sample, pred_dict = DMO_optimization(query, image, EBM, DMO, intrinsic, max_iteration=iteration, threshold=threshold, sort=True, verbose=True, keep_sample=True, limit_sample=limit_sample, device=device)

    elif sample_method == "sort":
        sample, pred_dict = sort_sample(query, image, EBM, device)
    
    else:
        raise ValueError("invalid inference method")

    return sample, pred_dict

def get_statistics_info(dataset):
    original_rot_mode = dataset.rot_mode
    dataset.rot_mode = "euler"
    temp_dict = {}
    print("get statistics")
    for i in tqdm(range(len(dataset))):
        image, h_query = dataset[i]

        for key in h_query.keys():
            if key not in temp_dict.keys():
                temp_dict[key] = [h_query[key]]
            else:
                temp_dict[key].append(h_query[key])

    info_dict = {}
    for key in temp_dict.keys():
        temp_dict[key] = torch.stack(temp_dict[key])
        max_value, _ = torch.max(temp_dict[key], 0)
        min_value, _ = torch.min(temp_dict[key], 0)
        mean = torch.mean(temp_dict[key], 0)
        std = torch.std(temp_dict[key], 0)

        info_dict[f"{key}_max"] = max_value
        info_dict[f"{key}_min"] = min_value
        info_dict[f"{key}_mean"] = mean
        info_dict[f"{key}_std"] = std
    dataset.rot_mode = original_rot_mode
    return info_dict

class Sampler():

    def __init__(self, first_sample, second_sample, cfg, info_dict, intrinsic='none', rot_mode="6d", frame=100, vae="none"):
        self.first_sample = first_sample
        self.second_sample = second_sample
        self.intrinsic = intrinsic
        self.rot_mode = rot_mode
        self.cfg = cfg
        self.info_dict = info_dict

        if "vae" in first_sample:
            if type(vae) == str:
                if rot_mode == "6d":
                    rot_dim = 6
                elif rot_mode == "quat":
                    rot_dim = 4

                input_size = (2 + 1 + rot_dim + 1) * (frame + 1)

                if cfg.VAE.NAME == "VAE":
                    vae = VAE(input_size, cfg.VAE.LATENT_DIM, intrinsic=intrinsic)
                elif cfg.VAE.NAME == "Transformer_VAE":
                    vae = Single_Class_TransformerVAE(["uv","z","rotation","grasp_state"],[2,1,rot_dim,1], frame + 1, latent_dim=cfg.VAE.LATENT_DIM, intrinsic=intrinsic)

                vae_pretrained_path = f"../global_result/RLBench/{cfg.DATASET.RLBENCH.TASK_NAME}/{cfg.VAE.NAME}_frame_{frame}_latentdim_{cfg.VAE.LATENT_DIM}_mode_{rot_mode}/model/model_iter100000.pth"
                if not os.path.exists(vae_pretrained_path):
                    raise ValueError(f"please train the vae first and save it at {vae_pretrained_path}")
                vae, _, _, _, _ = load_checkpoint(vae, vae_pretrained_path)

            if first_sample == "vae_noise":
                self.sampler = VAE_Noise_Sampler(vae)
            elif first_sample == "vae_and_random":
                self.sampler = VAE_Noise_and_RandomPick_Sampler(vae)
            elif first_sample == "vae_sample":
                self.sampler = VAE_Sampler(vae)
            elif first_sample == "vae_sample_and_random":
                self.sampler = VAE_Random_and_RandomPick_Sampler(vae)

    def __call__(self, num_sample, image, ebm_model='none', vae_model='none', sample_shape=[], pos_sample=None, device='cuda', verbose=False, valid_second_sample=True):
        # get sample shape
        if sample_shape != []:
            B, S = sample_shape[0], sample_shape[1]
        elif pos_sample != None:
            B, S, _ = pos_sample["pos"].shape
        else:
            raise ValueError("Please change sample_shape or pos_sample.")

        ### get first sample###
        if self.first_sample == "random_range":
            # get_sample
            negative_sample = random_negative_sample_RLBench([B,S], num_sample, self.info_dict, rot_mode="6d", device=device, z_range=[1.2,1.6])

        elif self.first_sample == "random_pick":
            # check
            if pos_sample == None:
                raise ValueError("pos_sample is required for random pick")
            # get sample
            negative_sample = get_negative_sample_from_batch(pos_sample, num_sample, include_self=False)

        elif self.first_sample == "random_pick_with_noise":
            # check
            if type(self.intrinsic) == str:
                raise ValueError("please change intrinsic")
            # config
            noise_range = self.cfg.SAMPLING.RPWN.RANGE
            include_self = self.cfg.SAMPLING.RPWN.INCLUDE_SELF
            # get sample
            negative_sample = random_dataset_with_noise(pos_sample, num_sample, self.info_dict, self.intrinsic, noise_range=noise_range, do_norm=True, image_size=(256,256), rot_mode=self.rot_mode, include_self=include_self)
        
        elif self.first_sample == "vae_sample":
            # get_sample
            negative_sample = self.sampler(num_sample, pos_sample)
        elif self.first_sample == "vae_noise":
            # config
            noise_std_mul = self.cfg.SAMPLING.VAE.NOISE_STD
            # get_sample
            negative_sample = self.sampler(num_sample, pos_sample, noise_std_mul=noise_std_mul)

        elif self.first_sample == "vae_and_random":
            # config
            noise_std_mul = self.cfg.SAMPLING.VAE.NOISE_STD
            # get_sample
            negative_sample = self.sampler(num_sample, pos_sample, noise_std_mul=noise_std_mul)
        
        elif self.first_sample == "vae_sample_and_random":
            # get_sample
            negative_sample = self.sampler(num_sample, pos_sample)

        else:
            raise ValueError("invalid first sample method")
            
        ### second sample ###
        if valid_second_sample:
            if self.second_sample == "langevin":
                # check
                if type(ebm_model) == str:
                    raise ValueError("please change model")
                    
                # config
                lr = self.cfg.SAMPLING.LANGEVIN.STEP_SIZE
                momentum = self.cfg.SAMPLING.LANGEVIN.MOMENTUM
                max_iteration = self.cfg.SAMPLING.LANGEVIN.ITERATION
                decay_step = self.cfg.SAMPLING.LANGEVIN.DECAY_RATIO
                decay_ratio = self.cfg.SAMPLING.LANGEVIN.DECAY_STEP
                
                # get sample
                negative_sample, _ = langevin(negative_sample, image, ebm_model, self.intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, verbose=verbose, sort=True)

            elif self.second_sample == "langevin_train":
                # check
                if type(ebm_model) == str:
                    raise ValueError("please change model")
                    
                # config
                lr = self.cfg.SAMPLING.LANGEVIN.STEP_SIZE
                momentum = self.cfg.SAMPLING.LANGEVIN.MOMENTUM
                max_iteration = self.cfg.SAMPLING.LANGEVIN.ITERATION
                decay_step = self.cfg.SAMPLING.LANGEVIN.DECAY_RATIO
                decay_ratio = self.cfg.SAMPLING.LANGEVIN.DECAY_STEP
                
                # get sample
                negative_sample, _ = langevin(negative_sample, image, ebm_model, self.intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, keep_half=True, verbose=verbose, sort=True)

            elif self.second_sample == "DFO":
                # check
                if type(ebm_model) == str:
                    raise ValueError("please change model")
                if type(self.intrinsic) == str:
                    raise ValueError("please change intrinsic")
                
                # config
                max_iteration = self.cfg.SAMPLING.DFO.ITERATION
                noise_range = self.cfg.SAMPLING.DFO.RATIO
                decay_step = self.cfg.SAMPLING.DFO.DECAY_RATIO
                decay_ratio = self.cfg.SAMPLING.DFO.DECAY_STEP
                
                # get sample
                negative_sample, _ = DFO(negative_sample, image, ebm_model, self.info_dict, self.intrinsic, max_iteration=max_iteration,
                        noise_range=noise_range, device=device, do_norm=True, sort=True, verbose=False)
            
            elif self.second_sample == "sgd":
                # config
                lr = self.cfg.SAMPLING.LANGEVIN.STEP_SIZE
                momentum = self.cfg.SAMPLING.LANGEVIN.MOMENTUM
                max_iteration = self.cfg.SAMPLING.LANGEVIN.ITERATION
                decay_step = self.cfg.SAMPLING.LANGEVIN.DECAY_RATIO
                decay_ratio = self.cfg.SAMPLING.LANGEVIN.DECAY_STEP

                # get sample
                negative_sample, _ = langevin(negative_sample, image, ebm_model, self.intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, noise=False, verbose=verbose, sort=True)

            elif self.second_sample == "langevin_vae":
                # check
                if type(ebm_model) == str:
                    raise ValueError("please change model")
                if type(vae_model) == str:
                    raise ValueError("please change vae")

                # config
                lr = self.cfg.SAMPLING.LANGEVIN_VAE.STEP_SIZE
                momentum = self.cfg.SAMPLING.LANGEVIN_VAE.MOMENTUM
                max_iteration = self.cfg.SAMPLING.LANGEVIN_VAE.ITERATION
                decay_step = self.cfg.SAMPLING.LANGEVIN_VAE.DECAY_RATIO
                decay_ratio = self.cfg.SAMPLING.LANGEVIN_VAE.DECAY_STEP

                # get sample
                negative_sample, _ = langevin_vae(negative_sample, image, ebm_model, vae_model, self.intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, noise=True, verbose=verbose, sort=True)
            
            elif self.second_sample == "langevin_vae_train":
                # check
                if type(ebm_model) == str:
                    raise ValueError("please change model")
                if type(vae_model) == str:
                    raise ValueError("please change vae")

                # config
                lr = self.cfg.SAMPLING.LANGEVIN_VAE.STEP_SIZE
                momentum = self.cfg.SAMPLING.LANGEVIN_VAE.MOMENTUM
                max_iteration = self.cfg.SAMPLING.LANGEVIN_VAE.ITERATION
                decay_step = self.cfg.SAMPLING.LANGEVIN_VAE.DECAY_RATIO
                decay_ratio = self.cfg.SAMPLING.LANGEVIN_VAE.DECAY_STEP

                # get sample
                negative_sample, _ = langevin_vae(negative_sample, image, ebm_model, vae_model, self.intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device=device, keep_half=True, noise=True, verbose=verbose, sort=False)
        
        return negative_sample

def random_negative_sample_RLBench(query_shape, num_sample, info_dict, rot_mode="6d", device='cuda', z_range=[1.2,1.6]):
    """
    pos_query:
     - uv
     - rotation_quat
     - grasp
     - z
     - time
    """
    negative_query = {}
    
    if len(query_shape) == 2:
        B, S = query_shape[0], query_shape[1]
        sample_shape = [B, num_sample, S, 1]
        num_sample = B * S * num_sample
    elif len(query_shape) == 1:
        S = query_shape[0]
        ample_shape = [num_sample, S, 1]
        num_sample = S * num_sample
        
    # get uv
    negative_u = (torch.rand(sample_shape[:-1], device=device) * 2) - 1
    negative_v = (torch.rand(sample_shape[:-1], device=device) * 2) - 1
    negative_uv = torch.stack([negative_u, negative_v], dim=len(sample_shape)-1)
    # negative_uv = rearrange(negative_uv, '(B N) P -> B N P', B=B)
    negative_query["uv"] = negative_uv
    
    # get rotation_quat
    if rot_mode == "quat":
        negative_rotation = R.random(num_sample).as_quat()
        negative_rotation = torch.tensor(negative_rotation, dtype=torch.float, device=device)
    elif rot_mode == "6d":
        negative_rotation = R.random(num_sample).as_matrix()
        negative_rotation = pytorch3d.transforms.matrix_to_rotation_6d(torch.tensor(negative_rotation, dtype=torch.float))
    else:
        raise ValueError("invalid rot_mode")
    
    if len(query_shape) == 2:
        negative_rotation = rearrange(negative_rotation, '(B N S) P -> B N S P', B=B, S=S)
    elif len(query_shape) == 1:
        negative_rotation = rearrange(negative_rotation, '(N S) P -> N S P', S=S)
    negative_query["rotation_quat"] = negative_rotation
    
    # get grasp
    negative_grasp = torch.randint(0, 2, sample_shape, dtype=torch.float, device=device)
    negative_query["grasp"] = negative_grasp
    
    # get z
    min_z, max_z = z_range[0], z_range[1]
    negative_z = torch.rand(sample_shape, device=device) * (max_z - min_z) + min_z
    negative_query["z"] = negative_z
    
    return negative_query

def get_negative_sample_from_batch(query, num_sample, include_self=False):
    B, S, N = query["uv"].shape
    
    if include_self:
        index_prob_matrix = torch.ones(B, B)
    else:
        index_prob_matrix = torch.ones(B, B) - torch.eye(B)
        
    sampling_index = torch.multinomial(index_prob_matrix, num_sample, replacement=True)

    negative_query = {}
    for key in query.keys():
        negative_query[key] = query[key][sampling_index]
    return negative_query

class VAE_Sampler():
    
    def __init__(self, model):
        
        self.model = model
        self.model.eval()
        
    def __call__(self, num_sample, h_query, device="cuda"):
        self.model = self.model.to(device)
        B = h_query["pos"].shape[0]
        
        for key in h_query.keys():
            h_query[key] = h_query[key].to(device)

        with torch.no_grad():
            negative_query = self.model.sample(B * num_sample, device)
            
        for key in negative_query.keys():
            negative_query[key] = rearrange(negative_query[key], "(B N) S D -> B N S D", N=num_sample)
            
        negative_query["time"] = repeat(h_query["time"], "B S D -> B N S D", N=num_sample)
        
        return negative_query

class VAE_Random_and_RandomPick_Sampler():
    
    def __init__(self, model):
            
        self.model = model
        self.model.eval()
        
    def __call__(self, num_sample, h_query, device="cuda"):
        self.model = self.model.to(device)
        B = h_query["pos"].shape[0]

        for key in h_query.keys():
            h_query[key] = h_query[key].to(device)

        num_random_sample = int(num_sample / 2)
        num_vae_sample = num_sample - num_random_sample

        random_negative_sample = get_negative_sample_from_batch(h_query, num_random_sample, include_self=False)

        with torch.no_grad():
            vae_negative_query = self.model.sample(B * num_vae_sample, device)
            
        for key in vae_negative_query.keys():
            vae_negative_query[key] = rearrange(vae_negative_query[key], "(B N) S D -> B N S D", N=num_vae_sample)
            
        vae_negative_query["time"] = repeat(h_query["time"], "B S D -> B N S D", N=num_vae_sample)
        
        negative_query = {}
        for key in vae_negative_query.keys():
            negative_query[key] = torch.cat([random_negative_sample[key], vae_negative_query[key]], 1).to(device)

        return negative_query

class VAE_Noise_Sampler():
    
    def __init__(self, model):
        
        self.model = model
        self.model.eval()
        
    def __call__(self, num_sample, h_query, noise_std_mul=1.0, device="cuda"):
        self.model = self.model.to(device)
        B = h_query["pos"].shape[0]
        
        for key in h_query.keys():
            h_query[key] = h_query[key].to(device)

        with torch.no_grad():
            negative_query, _ = self.model.sample_from_query(h_query, num_sample, noise_std=noise_std_mul)
            index_list = list(range(B * num_sample))
            random.shuffle(index_list)

            for key in negative_query.keys():
                negative_query[key] = rearrange(negative_query[key], "B N S D -> (B N) S D")
                negative_query[key] = negative_query[key][index_list]
            
        for key in negative_query.keys():
            negative_query[key] = rearrange(negative_query[key], "(B N) S D -> B N S D", N=num_sample)
            
        negative_query["time"] = repeat(h_query["time"], "B S D -> B N S D", N=num_sample)
        
        return negative_query

class VAE_Noise_and_RandomPick_Sampler():
    
    def __init__(self, model):
            
        self.model = model
        self.model.eval()
        
    def __call__(self, num_sample, h_query, noise_std_mul=1.0, device="cuda"):
        self.model = self.model.to(device)
        B = h_query["pos"].shape[0]

        for key in h_query.keys():
            h_query[key] = h_query[key].to(device)

        num_random_sample = int(num_sample / 2)
        num_vae_sample = num_sample - num_random_sample

        random_negative_sample = get_negative_sample_from_batch(h_query, num_random_sample, include_self=False)

        with torch.no_grad():
            vae_negative_query, _ = self.model.sample_from_query(h_query, num_vae_sample, noise_std=noise_std_mul)
            index_list = list(range(B * num_vae_sample))
            random.shuffle(index_list)

            for key in vae_negative_query.keys():
                vae_negative_query[key] = rearrange(vae_negative_query[key], "B N S D -> (B N) S D")
                vae_negative_query[key] = vae_negative_query[key][index_list]
            
        for key in vae_negative_query.keys():
            vae_negative_query[key] = rearrange(vae_negative_query[key], "(B N) S D -> B N S D", N=num_vae_sample)
            
        vae_negative_query["time"] = repeat(h_query["time"], "B S D -> B N S D", N=num_vae_sample)
        
        negative_query = {}
        for key in vae_negative_query.keys():
            negative_query[key] = torch.cat([random_negative_sample[key], vae_negative_query[key]], 1).to(device)

        return negative_query

def random_dataset_with_noise(query, num_sample, info_dict, intrinsic, noise_range=0.1, do_norm=True, image_size=(256,256), rot_mode="6d", include_self=False):
    negative_query = get_negative_sample_from_batch(query, num_sample, include_self=include_self)
    negative_query = add_noise_for_negative(negative_query, info_dict, intrinsic, noise_range, do_norm, image_size, rot_mode)
    return negative_query

def add_noise_for_negative(negative_query, info_dict, intrinsic, noise_range=0.1, do_norm=True, image_size=(256,256), rot_mode="6d"):
    # pos
    pos = negative_query["pos"]
    device = negative_query["pos"].device
    B, N, S, _ = pos.shape
    rand_pos = torch.rand(B, N, S, 3) * 2 - 1
    max_pos, min_pos = info_dict["pos_max"], info_dict["pos_min"]
    range_pos = (max_pos - min_pos) * noise_range / 2
    rand_pos = rand_pos * range_pos
    noise_pos = pos + rand_pos.to(device)
    negative_query["pos"] = noise_pos
    
    # uv and z
    noise_z = noise_pos[:,:,:,2]
    z_repet = repeat(noise_z, "B N S -> B N S Z", Z=3)
    pos_data = noise_pos / z_repet # u,v,1
    noise_uv = torch.einsum('ij,bnsj->bnsi', torch.tensor(intrinsic, dtype=torch.float, device=device), pos_data)
    u, v = noise_uv[:, :, :, 0], noise_uv[:, :, :, 1]
    noise_z = torch.unsqueeze(noise_z, 3)

    if do_norm == True:
        h, w = image_size
        u = (u / (w - 1) * 2) - 1
        v = (v / (h - 1) * 2) - 1
        
        noise_uv = torch.stack([u, v], 3)
    negative_query["uv"] = noise_uv.to(device)
    negative_query["z"] = noise_z.to(device)
            
    # rot
    rot = negative_query["rotation"]
    rand_rot = torch.rand(B, N, S, 3) * 2 - 1
    max_rot, min_rot = info_dict["rotation_max"], info_dict["rotation_min"]
    range_rot = (max_rot - min_rot) * noise_range / 2
    rand_rot = rand_rot * range_rot
    rand_rot = rearrange(rand_rot, "B N S D -> (B N S) D")
    if rot_mode == "quat":
        rot = rearrange(rot, "B N S D -> (B N S) D")
        rot_r = R.from_quat(rot.cpu().numpy())
        rot_euler = rot_r.as_euler('zxy', degrees=True)
        noise_rot_euler = rot_euler + rand_rot.numpy()
        noise_rot_r = R.from_euler('zxy', noise_rot_euler, degrees=True)
        noise_rot = noise_rot_r.as_quat()
        noise_rot = torch.tensor(noise_rot, dtype=torch.float)
        noise_rot = rearrange(noise_rot, "(B N S) D -> B N S D", B=B, N=N)
    elif rot_mode == "6d":
        rot = rearrange(rot, "B N S D -> (B N S) D")
        rot = pytorch3d.transforms.rotation_6d_to_matrix(rot)
        rot_r = R.from_matrix(rot.cpu().numpy())
        rot_euler = rot_r.as_euler('zxy', degrees=True)
        noise_rot_euler = rot_euler + rand_rot.numpy()
        noise_rot_r = R.from_euler('zxy', noise_rot_euler, degrees=True)
        noise_rot = noise_rot_r.as_matrix()
        noise_rot = torch.tensor(noise_rot, dtype=torch.float)
        noise_rot = pytorch3d.transforms.matrix_to_rotation_6d(noise_rot)
        noise_rot = rearrange(noise_rot, "(B N S) D -> B N S D", B=B, N=N)
    negative_query["rotation"] = noise_rot.to(device)

    # grasp
    grasp = negative_query["grasp_state"]
    grasp_mean = info_dict["grasp_state_mean"]
    grasp_round = torch.round(info_dict["grasp_state_mean"])
    prob = torch.abs(grasp_round - grasp_mean)
    ones = torch.bernoulli(prob) * -1
    noise_grasp = torch.abs(grasp - ones.to(device)) 
    negative_query["grasp_state"] = noise_grasp.to(device)
    
    return negative_query

def langevin(query, image, model, intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device='cuda', noise=True, verbose=False, sort=True, get_pose=True, keep_half=False, image_size=(256,256), keep_sample=False, limit_sample=0, energy_limit=1e+3):
    query = copy.deepcopy(query)
    B,N,S,_ = query["pos"].shape

    model = model.to(device)
    model.eval()

    image = image.to(device)
    for key in query.keys():
        query[key] = query[key].to(device)

    # keep some query
    if keep_half:
        # dont use this for inference
        keep_num = int(N / 2)
        
        keep_query = {}
        for key in query.keys():
            keep_query[key] = query[key][:,:keep_num]
            query[key] = query[key][:,keep_num:]
    
    query.pop("pos")

    for param in model.enc.parameters():
        param.requires_grad = False

    for param in model.dec.parameters():
        param.requires_grad = False

    with torch.no_grad():
        score_dict, _ = model(image, query)

    # extract n sample
    if limit_sample != 0:

        _, indices = torch.sort(score_dict['score'], 1)

        for key in score_dict.keys():
            score_dict[key] = torch.gather(score_dict[key], 1, indices)
            score_dict[key] = score_dict[key][:,:limit_sample]

        for key in query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
            query[key] = torch.gather(query[key], 1, new_indices)
            query[key] = query[key][:,:limit_sample]
    
    # for keep query
    final_query = {}
    final_score = {}
    
    if keep_sample:
        for key in query.keys():
            final_query[key] = copy.deepcopy(query[key].cpu())

        for key in score_dict.keys():
            final_score[key] = score_dict[key].cpu()

    for key in query.keys():
        if key in ["time", "pos"]:
            continue
        query[key].requires_grad = True

    query_optimizer = torch.optim.SGD(query.values(), lr=lr, momentum=momentum)
    query_scheduler = torch.optim.lr_scheduler.StepLR(query_optimizer, step_size=decay_step, gamma=decay_ratio)

    # optimize query
    if max_iteration > 0:
        for iteration in range(max_iteration):
            query_optimizer.zero_grad()

            score_dict, _ = model(image, query, with_feature=True)
            mean_energy = torch.mean(score_dict['score'])

            if verbose:
                print(f"iteration: {iteration}")
                print("min energy")
                print(torch.mean(torch.min(score_dict["score"], 1)[0]))
                print("mean energy")
                print(torch.mean(score_dict["score"]))

            # check loss
            if torch.isnan(mean_energy):
                pass
            elif abs(mean_energy) > energy_limit:
                print("Energy is over the theresholds")
                print(f"mean energy: {torch.mean(score_dict['score'])}")
                break
            else:
                mean_energy.backward()
                query_optimizer.step()
                query_scheduler.step()

            lr = query_optimizer.param_groups[0]['lr']

            with torch.no_grad():
                if noise:
                    var = 2 * lr # see https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
                    for key in query.keys():
                        B,N,S,D = query[key].shape
                        query[key] = query[key] + torch.normal(mean=0., std=torch.ones(B,N,S,D) * var).to(device)

                query = clipping(query)
            
            if keep_sample and (iteration < max_iteration - 1):
                for key in query.keys():
                    final_query[key] = torch.cat([final_query[key], copy.deepcopy(query[key].detach().cpu())], 1)

                for key in score_dict.keys():
                    final_score[key] = torch.cat([final_score[key], score_dict[key].cpu()], 1)

    for param in model.enc.parameters():
        param.requires_grad = True

    for param in model.dec.parameters():
        param.requires_grad = True

    with torch.no_grad():
        score_dict, _ = model(image, query, with_feature=True)
    
    for key in query.keys():
        if key in final_query.keys():
            final_query[key] = torch.cat([final_query[key], copy.deepcopy(query[key].detach().cpu())], 1)
        else:
            final_query[key] = query[key].detach().cpu()

    for key in score_dict.keys():
        if key in final_score.keys():
            final_score[key] = torch.cat([final_score[key], score_dict[key].cpu()], 1)
        else:
            final_score[key] = score_dict[key].cpu()

    if sort:
        _, indices = torch.sort(final_score['score'], 1)
        for key in final_score.keys():
            final_score[key] = torch.gather(final_score[key], 1, indices)

        for key in final_query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=final_query[key].shape[-1])
            final_query[key] = torch.gather(final_query[key], 1, new_indices)

    if verbose:
        print(f"final min score: {torch.min(final_score['score'])}")

    if get_pose:
        uv = final_query["uv"]
        z = final_query["z"]
        h, w = image_size
        device = uv.device

        u, v = uv[:,:,:,0], uv[:,:,:,1]
        u = (u + 1) / 2 * (w - 1)
        v = (v + 1) / 2 * (h - 1)
        uv_denorm = torch.stack([u, v], 3)
        ones = torch.ones(*uv_denorm.shape[:-1], 1).to(device)
        uv_denorm = torch.cat([uv_denorm, ones], 3)
        intrinsic = torch.tensor(intrinsic, dtype=torch.float).to(device)
        inv_intrinsic = torch.linalg.inv(intrinsic)

        xy = torch.einsum('ij,bnkj->bnki', inv_intrinsic, uv_denorm)
        xyz = xy * z
        final_query["pos"] = xyz
    
    if keep_half:
        for key in final_query.keys():
            final_query[key] = torch.cat([keep_query[key].to(device), final_query[key].to(device)], 1)
            
    model.train()

    return final_query, final_score

def DFO(query, image, model, info_dict, intrinsic, max_iteration=20, noise_range=0.1, do_norm=True, image_size=(256,256), 
        decay_step=10, decay_ratio=0.5, device='cuda', sort=True, verbose=False):

    model = model.to(device)
    model.eval()

    for key in query:
        query[key] = query[key].to(device)
    image = image.to(device)

    m = torch.nn.Softmax(dim=1)
    B,N,S,_ = query["pos"].shape

    for iteration in range(1, max_iteration+1):
        with torch.no_grad():
            if iteration == 1:
                pred_dict, _ = model(image, query)
            else:
                pred_dict, _ = model(image, query, with_feature=True)

            prob = m(-pred_dict["score"])
            index = torch.multinomial(prob, N, replacement=True)

            for key in query.keys():
                new_indices = repeat(index, 'B N -> B N S D', S=S, D=query[key].shape[-1])
                query[key] = torch.gather(query[key], 1, new_indices)
            
            if iteration % decay_step == 0:
                noise_range *= decay_ratio
            query = add_noise_for_negative(query, info_dict, intrinsic, noise_range, do_norm, image_size)

    with torch.no_grad():
        if max_iteration:
            pred_dict, _ = model(image, query, with_feature=True)
        else:
            pred_dict, _ = model(image, query, with_feature=False)

    model.train()

    if sort:
        _, indices = torch.sort(pred_dict['score'], 1)
        for key in pred_dict.keys():
            pred_dict[key] = torch.gather(pred_dict[key], 1, indices)

        for key in query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
            query[key] = torch.gather(query[key], 1, new_indices)

    if verbose:
        print("result")
        print("energy")
        print(pred_dict['score'][:,:5])
        print("min energy")
        print(torch.mean(torch.min(pred_dict["score"], 1)[0]))
        print("mean energy")
        print(torch.mean(pred_dict["score"]))
    return query, pred_dict

def sort_sample(query, image, model, device):
    B,N,S,_ = query["pos"].shape
    
    for key in query.keys():
        query[key] = query[key].to(device)
    image = image.to(device)
    model = model.to(device)
    
    
    model.eval()
    with torch.no_grad():
        pred_dict, _ = model(image, query)
    model.train()

    _, indices = torch.sort(pred_dict['score'], 1)
    for key in pred_dict.keys():
        pred_dict[key] = torch.gather(pred_dict[key], 1, indices)

    for key in query.keys():
        new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
        query[key] = torch.gather(query[key], 1, new_indices)
    
    return query, pred_dict

def langevin_vae(query, image, EBM, vae, intrinsic, lr, momentum, max_iteration, decay_step, decay_ratio, device='cuda', noise=True, verbose=False, sort=True, get_pose=True, keep_half=False, image_size=(256,256), keep_sample=False, limit_sample=0):
    query = copy.deepcopy(query)
    B,N,S,_ = query["pos"].shape
    EBM = EBM.to(device)
    EBM.eval()

    for param in EBM.enc.parameters():
        param.requires_grad = False

    for param in EBM.dec.parameters():
        param.requires_grad = False
    
    image = image.to(device)
    vae.to(device)
    vae.eval()

    for key in query.keys():
        query[key] = query[key].to(device)

    # keep some query
    if keep_half:
        # dont use this for inference
        keep_num = int(N / 2)
        
        keep_query = {}
        for key in query.keys():
            keep_query[key] = query[key][:,:keep_num]
            query[key] = query[key][:,keep_num:]

    # remove pos for concatenation
    # if "pos" in query.keys():
    #     query.pop("pos")
    
    with torch.no_grad():
        score_dict, _ = EBM(image, query)

    # extract n sample
    if limit_sample != 0:
        _, indices = torch.sort(score_dict['score'], 1)

        for key in score_dict.keys():
            score_dict[key] = torch.gather(score_dict[key], 1, indices)
            score_dict[key] = score_dict[key][:,:limit_sample]

        for key in query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
            query[key] = torch.gather(query[key], 1, new_indices)
            query[key] = query[key][:,:limit_sample]
    
    # for keep query
    final_query = {}
    final_score = {}
    if keep_sample:
        for key in query.keys():
            final_query[key] = query[key]

        for key in score_dict.keys():
            final_score[key] = score_dict[key]

    if verbose:
        print(f"first min score: {torch.min(score_dict['score'])}")

    # change shape for VAE
    for key in query.keys():
        query[key] = rearrange(query[key], 'B N S D -> (B N) S D')

    time = query["time"]

    # get z
    with torch.no_grad():
        z = vae.encode(query)
    
    z = z.detach()
    z.requires_grad = True
    z_optimizer = torch.optim.SGD([z], lr=lr, momentum=momentum)
    z_scheduler = torch.optim.lr_scheduler.StepLR(z_optimizer, step_size=decay_step, gamma=decay_ratio)
    base_distribution = torch.distributions.normal.Normal(torch.zeros(vae.latent_dim).to(device), torch.ones(vae.latent_dim).to(device))
    
    # optimize
    for iteration in range(max_iteration):
        z_optimizer.zero_grad()

        query = vae.decode(z)
        query["time"] = copy.deepcopy(time)
        
        for key in query.keys():
            query[key] = rearrange(query[key], '(B N) S D -> B N S D',B=B)

        if keep_sample and (iteration < max_iteration - 1):
            for key in query.keys():
                final_query[key] = torch.cat([final_query[key], copy.deepcopy(query[key].detach())], 1)

            for key in score_dict.keys():
                final_score[key] = torch.cat([final_score[key], score_dict[key]], 1)

        score_dict, _ = EBM(image, query, with_feature=True)

        base_energy = torch.sum(base_distribution.log_prob(z), 1)
        base_energy = rearrange(base_energy, "(B N) -> B N", B=B)
        energy = score_dict['score'] - base_energy
        mean_energy = torch.mean(energy)

        if verbose:
            print(f"iteration: {iteration}")
            print(f"min energy: {torch.mean(torch.min(score_dict['score'], 1)[0]).detach().item()}")
            print(f"mean energy: {torch.mean(score_dict['score'])}")
            print(f"max_energy: {torch.mean(torch.max(score_dict['score'], 1)[0]).detach().item()}")

        if mean_energy > 1e+4:
            print("mean energy over thereshold")
            break

        mean_energy.backward()
        torch.nn.utils.clip_grad_norm_([z], 0.01, norm_type=2.0)
        z_optimizer.step()
        z_scheduler.step()

        lr = z_optimizer.param_groups[0]['lr']

        with torch.no_grad():
            if noise:
                var = 2 * lr * 0.001# see https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
                z_shape = z.shape
                z  = z + torch.normal(mean=0., std=torch.ones(z_shape) * var).to(device)
    
    for key in query.keys():
        if key in final_query.keys():
            final_query[key] = torch.cat([final_query[key], copy.deepcopy(query[key].detach())], 1)
        else:
            final_query[key] = query[key].detach()

    for key in score_dict.keys():
        if key in final_score.keys():
            final_score[key] = torch.cat([final_score[key].to(device), score_dict[key].to(device)], 1)
        else:
            final_score[key] = score_dict[key]

    for param in EBM.enc.parameters():
        param.requires_grad = True

    for param in EBM.dec.parameters():
        param.requires_grad = True

    if sort:
        _, indices = torch.sort(final_score['score'], 1)
        for key in final_score.keys():
            final_score[key] = torch.gather(final_score[key], 1, indices)

        for key in final_query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=final_query[key].shape[-1])
            final_query[key] = torch.gather(final_query[key], 1, new_indices)

    if verbose:
        print(f"final min score: {torch.min(final_score['score'])}")

    if get_pose:
        uv = final_query["uv"]
        z = final_query["z"]
        h, w = image_size
        device = uv.device

        u, v = uv[:,:,:,0], uv[:,:,:,1]
        u = (u + 1) / 2 * (w - 1)
        v = (v + 1) / 2 * (h - 1)
        uv_denorm = torch.stack([u, v], 3)
        ones = torch.ones(*uv_denorm.shape[:-1], 1).to(device)
        uv_denorm = torch.cat([uv_denorm, ones], 3)
        intrinsic = torch.tensor(intrinsic, dtype=torch.float).to(device)
        inv_intrinsic = torch.linalg.inv(intrinsic)

        xy = torch.einsum('ij,bnkj->bnki', inv_intrinsic, uv_denorm)
        xyz = xy * z
        final_query["pos"] = xyz
    
    if keep_half:
        for key in final_query.keys():
            final_query[key] = torch.cat([keep_query[key].to(device), final_query[key].to(device)], 1)

    EBM.train()

    return final_query, final_score

def DMO_optimization(query, image, EBM, DMO, intrinsic, max_iteration=10, threshold=-1000, sort=True, verbose=False, get_pose=True, image_size=(256,256), keep_sample=True, limit_sample=16, device="cpu"):
    EBM = EBM.to(device)
    DMO = DMO.to(device)
    
    EBM.eval()
    DMO.eval()

    for key in query:
        query[key] = query[key].to(device)
    image = image.to(device)

    _,N,S,_ = query["pos"].shape
    
    if "pos" in query.keys():
        query.pop("pos")
    
    with torch.no_grad():
        score_dict, _ = EBM(image, query)

    if verbose:
        print(f"iteration: 0 min score: {torch.min(score_dict['score'])}")

    # extract n sample
    if limit_sample != 0:
        _, indices = torch.sort(score_dict['score'], 1)

        for key in score_dict.keys():
            score_dict[key] = torch.gather(score_dict[key], 1, indices)
            score_dict[key] = score_dict[key][:,:limit_sample]

        for key in query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
            query[key] = torch.gather(query[key], 1, new_indices)
            query[key] = query[key][:,:limit_sample]

    time = query["time"]
    
    # for keep query
    final_query = {}
    final_score = {}

    if keep_sample:
        for key in query.keys():
            final_query[key] = query[key].to(device)

        for key in score_dict.keys():
            final_score[key] = score_dict[key].to(device)

    for iteration in range(1, max_iteration+1):
        with torch.no_grad():
                
            query, info = DMO(image, query)
            query["time"] = time
                
            if iteration == 1:
                score_dict, _ = EBM(image, query)
            else:
                score_dict, _ = EBM(image, query, with_feature=True)
            
            if verbose:
                print(f"iteration: {iteration} min score: {torch.min(score_dict['score'])}")
            
            if keep_sample or (iteration == max_iteration):
                for key in query.keys():
                    if key in final_query.keys():
                        final_query[key] = torch.cat([final_query[key], query[key]], 1)
                    else:
                        final_query[key] = query[key]

                for key in score_dict.keys():
                    if key in final_score.keys():
                        final_score[key] = torch.cat([final_score[key], score_dict[key]], 1)
                    else:
                        final_score[key] = score_dict[key]

            if torch.min(score_dict["score"]) < threshold:
                print("break loop")
                break
    
    if sort:
        _, indices = torch.sort(final_score['score'], 1)
        for key in final_score.keys():
            final_score[key] = torch.gather(final_score[key], 1, indices)

        for key in final_query.keys():
            new_indices = repeat(indices, 'B N -> B N S D', S=S, D=final_query[key].shape[-1])
            final_query[key] = torch.gather(final_query[key], 1, new_indices)

    if verbose:
        print(f"final min score: {torch.min(final_score['score'])}")

    if get_pose:
        uv = final_query["uv"]
        z = final_query["z"]
        h, w = image_size
        device = uv.device

        u, v = uv[:,:,:,0], uv[:,:,:,1]
        u = (u + 1) / 2 * (w - 1)
        v = (v + 1) / 2 * (h - 1)
        uv_denorm = torch.stack([u, v], 3)
        ones = torch.ones(*uv_denorm.shape[:-1], 1).to(device)
        uv_denorm = torch.cat([uv_denorm, ones], 3)
        intrinsic = torch.tensor(intrinsic, dtype=torch.float).to(device)
        inv_intrinsic = torch.linalg.inv(intrinsic)

        xy = torch.einsum('ij,bnkj->bnki', inv_intrinsic, uv_denorm)
        xyz = xy * z
        final_query["pos"] = xyz
    
    return final_query, final_score

# def DMO_optimization(query, image, EBM, DMO, intrinsic, max_iteration=10, threshold=-10, sort=True, verbose=False, get_pose=True, image_size=(256,256), keep_good_sample=False, limit_sample=16, device="cpu"):
#     EBM = EBM.to(device)
#     DMO = DMO.to(device)
    
#     EBM.eval()
#     DMO.eval()

#     for key in query:
#         query[key] = query[key].to(device)
#     image = image.to(device)

#     _,N,S,_ = query["pos"].shape
    
#     with torch.no_grad():
#         score_dict, _ = EBM(image, query)

#     if verbose:
#         print(f"iteration: 0 min score: {torch.min(score_dict['score'])}")

#     # extract n sample
#     if limit_sample != 0:
#         N = limit_sample
#         _, indices = torch.sort(score_dict['score'], 1)

#         for key in score_dict.keys():
#             score_dict[key] = torch.gather(score_dict[key], 1, indices)
#             score_dict[key] = score_dict[key][:,:N]

#         for key in query.keys():
#             new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
#             query[key] = torch.gather(query[key], 1, new_indices)
#             query[key] = query[key][:,:N]

#     time = query["time"]
    
#     # for keep query
#     final_query = {}
#     final_score = {}
#     for key in query.keys():
#         final_query[key] = query[key].cpu()
    
#     for key in score_dict.keys():
#         final_score[key] = score_dict[key].cpu()

#     for iteration in range(1, max_iteration+1):
#         with torch.no_grad():
                
#             new_query, info = DMO(image, query)
#             new_query["time"] = time
                
#             if iteration == 1:
#                 new_score_dict, _ = EBM(image, new_query)
#             else:
#                 new_score_dict, _ = EBM(image, new_query, with_feature=True)
            
#             if verbose:
#                 print(f"iteration: {iteration} min score: {torch.min(new_score_dict['score'])}")
            
#             if keep_good_sample:
#                 for key in new_score_dict.keys():
#                     score_dict[key] = torch.cat([score_dict[key], new_score_dict[key]], 1)
                
#                 if "pos" in query.keys():
#                     query.pop("pos")

#                 for key in query.keys():
#                     query[key] = torch.cat([query[key], new_query[key]], 1)

#                 # sort
#                 _, indices = torch.sort(score_dict['score'], 1)
#                 for key in score_dict.keys():
#                     score_dict[key] = torch.gather(score_dict[key], 1, indices)
#                     score_dict[key] = score_dict[key][:,:N]

#                 for key in query.keys():
#                     new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
#                     query[key] = torch.gather(query[key], 1, new_indices)
#                     query[key] = query[key][:,:N]
#             else:
#                 score_dict = new_score_dict
#                 query = new_query


#             if torch.min(score_dict["score"]) < threshold:
#                 print("break loop")
#                 break
    
#     if sort:
#         _, indices = torch.sort(score_dict['score'], 1)
#         for key in score_dict.keys():
#             score_dict[key] = torch.gather(score_dict[key], 1, indices)

#         for key in query.keys():
#             new_indices = repeat(indices, 'B N -> B N S D', S=S, D=query[key].shape[-1])
#             query[key] = torch.gather(query[key], 1, new_indices)
            
#     if get_pose:
#         uv = query["uv"]
#         z = query["z"]
#         h, w = image_size
#         device = uv.device

#         u, v = uv[:,:,:,0], uv[:,:,:,1]
#         u = (u + 1) / 2 * (w - 1)
#         v = (v + 1) / 2 * (h - 1)
#         uv_denorm = torch.stack([u, v], 3)
#         ones = torch.ones(*uv_denorm.shape[:-1], 1).to(device)
#         uv_denorm = torch.cat([uv_denorm, ones], 3)
#         intrinsic = torch.tensor(intrinsic, dtype=torch.float).to(device)
#         inv_intrinsic = torch.linalg.inv(intrinsic)

#         xy = torch.einsum('ij,bnkj->bnki', inv_intrinsic, uv_denorm)
#         xyz = xy * z
#         query["pos"] = xyz
    
#     return query, score_dict

def random_relative_negative_sample_RLBench(cfg, base_uv, base_z, base_rot, info_dict, batch_size=0, device='cuda'):
    """
    pos_query:
     - uv
     - rotation_quat
     - grasp
     - z
     - time
    """
    base_uv, base_z, base_rot = base_uv.to(device), base_z.to(device), base_rot.to(device)
    max_z, min_z, max_uv, min_uv, max_rot, min_rot, max_time = info_dict["max_z_frame"], info_dict["min_z_frame"], info_dict["max_uv_frame"], info_dict["min_uv_frame"], info_dict["max_rot_frame"], info_dict["min_rot_frame"], info_dict["max_time"]
    
    num_query = cfg.SAMPLING.NUM_NEGATIVE
    negative_query = {}

    if batch_size != 0:
        sample_shape = [batch_size, num_query, 1]
        num_sample = cfg.SAMPLING.NUM_NEGATIVE * batch_size
    else:
        sample_shape = [num_query, 1]
        num_sample = cfg.SAMPLING.NUM_NEGATIVE

    
    # get uv
    max_u, max_v = max_uv
    min_u, min_v = min_uv
    u_range, v_range = max_u - min_u, max_v - min_v

    sample_min_u, sample_min_v = min_u - (u_range * 0.05), min_v - (v_range * 0.05)

    negative_u = torch.rand(sample_shape[:-1], device=device) * (u_range * 1.1) + sample_min_u
    negative_v = torch.rand(sample_shape[:-1], device=device) * (v_range * 1.1) + sample_min_v
    negative_uv_delta = torch.stack([negative_u, negative_v], dim=len(sample_shape)-1)
    negative_uv = base_uv + negative_uv_delta
    negative_query["uv"] = negative_uv
    
    # get rotation_quat
    if max_rot == None:
        negative_rotation = R.random(num_sample).as_quat()
        negative_rotation = torch.tensor(negative_rotation, dtype=torch.float, device=device)
        if batch_size != 0:
            negative_rotation = rearrange(negative_rotation, '(B N) P -> B N P', B=batch_size)
    
    rotation_range = torch.tensor(max_rot, dtype=torch.float, device=device) - torch.tensor(min_rot, dtype=torch.float, device=device)
    sample_rot_min = torch.tensor(min_rot, dtype=torch.float, device=device) - (rotation_range * 0.05)
    rotation_delta = torch.rand([*sample_shape[:-1], 3], dtype=torch.float, device=device) * (rotation_range * 1.1) + sample_rot_min
    rotation_delta_np = rotation_delta.cpu().numpy()
    r1 = R.from_euler('zyx', rotation_delta_np, degrees=True)
    r2 = R.from_quat(base_rot.cpu().numpy())
    r3 = r1 * r2
    negative_rotation = torch.tensor(r3.as_quat(), dtype=torch.float, device=device)
#     negative_rotation = rotation_delta * base_rot
    negative_query["rotation_quat"] = negative_rotation
    
    # get grasp
    negative_grasp = torch.randint(0, 2, sample_shape, dtype=torch.float, device=device)
    negative_query["grasp"] = negative_grasp
    
    # get z
    z_range = max_z - min_z
    sample_min_z = min_z - (z_range * 0.05)
    negative_z_delta = torch.rand(sample_shape, dtype=torch.float, device=device) * (z_range * 1.1) + sample_min_z
    negative_z = negative_z_delta + base_z
    negative_query["z"] = negative_z
    
#     # get time
#     if cfg.PRED_LEN == 0:
#         if max_time == None:
#             negative_time = torch.randint(1, 100, (num_sample,), dtype=torch.float16, device=device)
#         else:
#             negative_time = torch.randint(1, max_time, (num_sample,), dtype=torch.float16, device=device)
#     else:
#         time_list = torch.tensor(cfg.SAMPLING.TIME_LIST, device=device)
#         p = torch.ones(len(time_list), device=device) / len(time_list)
#         index = p.multinomial(num_samples=num_sample, replacement=True)
#         negative_time = time_list[index]
    
#     if batch_size != 0:
#         negative_time = rearrange(negative_time, '(B N) -> B N', B=batch_size)
    
#     negative_time = torch.unsqueeze(negative_time, len(sample_shape)-1) / 100
#     negative_query["time"] = negative_time.to(device)
    
    return negative_query

def clipping(query):
    keys = query.keys()
    
    if "uv" in keys:
        query["uv"] = torch.clip(query["uv"], -1, 1)
    
    if "grasp_state" in keys:
        query["grasp_state"] = torch.clip(query["grasp_state"], 0, 1)
    
    if "rotation" in keys:
        rot_dim = query["rotation"].shape[-1]

        if rot_dim != 6:
            raise ValueError("rotation clipping is valid for 6d representation")
        
        rotation = rearrange(query["rotation"], "B N S (A C) -> B N S A C", A=2)
        rotation = torch.nn.functional.normalize(rotation,dim=-1)
        query["rotation"] = rearrange(rotation, "B N S A C -> B N S (A C)", A=2)

    return query