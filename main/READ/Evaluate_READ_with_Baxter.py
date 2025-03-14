import os
import csv
import sys
import copy
import math
import json
import shutil
import datetime
import pickle
import argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from einops import rearrange, reduce, repeat

sys.path.append("../")

import torch
import torchvision
import torch.nn.functional as F

from pycode.dataset import Baxter_Demos
from pycode.config import _C as cfg
from pycode.misc import load_checkpoint, convert_rotation_6d_to_matrix, visualize_inf_query, get_pos, visualize_multi_query_pos
from pycode.misc import calculate_euclid_pos, calculate_euclid_angle, calculate_euclid_grasp, output2action, check_img, get_gt_pose, make_video, get_concat_h
from pycode.retrieval import Direct_Retrieval, Image_Based_Retrieval_SPE, BYOL_Retrieval, CLIP_Retrieval, MSE_Based_Retrieval
from pycode.READ.model import SPE_Continuous_Latent_Diffusion, Timm_Continuous_Latent_Diffusion,  AvgPool_Continuous_Latent_Diffusion, ConvPool_Continuous_Latent_Diffusion
from pycode.READ.vae import Single_Class_TransformerVAE
from pycode.READ import sampling, sde_lib, noise_sampler

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--vae_path', type=str, default="")
parser.add_argument('--inf_method', type=str, required=True) # random, reconstruct, retrieve_from_SPE, retrieve_from_CLIP, retrieve_from_BYOL_wo_crop, retrieve_from_SPE_wo_modification
parser.add_argument('--result_path', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--image_path', type=str, default="")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--config_path', type=str, default="../config/Test_LatentContinuousDiff.yaml")

args = parser.parse_args()

### SET CONFIG ###
input_keys = ["uv","z","rotation","grasp_state"]
input_dims = [2, 1, 6, 1]
rot_mode = "6d"
device = args.device
inference_method = args.inf_method

# load model config
model_path = args.model_path
model_config_path = os.path.join(model_path[:model_path.find("/model")], "Train_READ_Baxter.yaml")

print(f"model path: {model_path}")
print(f"model config path: {model_config_path}")
print("")

checkpoint_base_path = model_path[:model_path.find("/model")]
argfile_path = os.path.join(checkpoint_base_path, "args.json")
with open(argfile_path) as f:
    arg_info = json.load(f)

frame = arg_info["frame"]

################################################################################
### Configurations are changed to load the model and the sde.
################################################################################

cfg.merge_from_file(model_config_path)
cfg.DATASET.BAXTER.PATH = os.path.abspath('../dataset/baxter_demos')
train_dataset  = Baxter_Demos("train", cfg, save_dataset=True, num_frame=frame, rot_mode=rot_mode, keys=input_keys)

# set vae
if args.vae_path == "":
    vae_path = f"../weights/{cfg.DATASET.NAME}/ACTOR_frame_{frame}_latentdim_{cfg.VAE.LATENT_DIM}_KLD_{cfg.VAE.KLD_WEIGHT}/model/model_iter10000.pth"
else:
    vae_path = args.vae_path

vae = Single_Class_TransformerVAE(input_keys, input_dims, frame + 1, latent_dim=cfg.VAE.LATENT_DIM).to(device)
vae, _, _, _, _ = load_checkpoint(vae, vae_path)
vae.to(device)

# set model
model_name = cfg.MODEL.NAME
inout_type = cfg.MODEL.INOUT

conv_dims = cfg.MODEL.CONV_DIMS
enc_depths = cfg.MODEL.ENC_DEPTHS
enc_layers = cfg.MODEL.ENC_LAYERS

dec_depths = cfg.MODEL.DEC_DEPTHS
dec_layers = cfg.MODEL.DEC_LAYERS

extractor_name = cfg.MODEL.EXTRACTOR_NAME
predictor_name = cfg.MODEL.PREDICTOR_NAME

conv_droppath_rate = cfg.MODEL.CONV_DROP_PATH_RATE
query_emb_dim = cfg.MODEL.QUERY_EMB_DIM

vae_latent_dim = cfg.VAE.LATENT_DIM

if model_name == "Convnext-UNet":
    model = SPE_Continuous_Latent_Diffusion(input_keys, input_dims, vae, vae_latent_dim, inout_type,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate, input_dim=3)
elif model_name == "Convnext-UNet-avgpool":
    model = AvgPool_Continuous_Latent_Diffusion(input_keys, input_dims, vae, vae_latent_dim, inout_type,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate, input_dim=3)
elif model_name == "Convnext-UNet-convpool":
    model = ConvPool_Continuous_Latent_Diffusion(input_keys, input_dims, vae, vae_latent_dim, inout_type,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate, input_dim=3)
else:
    model = Timm_Continuous_Latent_Diffusion(input_keys, input_dims, model_name, vae, vae_latent_dim, inout_type, img_size=256, input_dim=3, pretrained=True,
                    query_emb_dim=query_emb_dim)
model = model.to(device)
                
print(model)
print("loading model")
model, _, _, _, _ = load_checkpoint(model, model_path)
model.to(device)
model.eval()

# set sde
sde_name = cfg.SDE.NAME
assert sde_name == 'irsde'
sde = sde_lib.IRSDE(keys=input_keys, lamda=cfg.SDE.IRSDES.LAMDA, theta_1=cfg.SDE.IRSDES.THETA, schedule="linear")
sampling_eps = 1e-3

# set noise sampler
noise_name = cfg.NOISE.NAME
if noise_name == "gaussian":
    noise_sample_func = noise_sampler.Normal_Noise()
elif noise_name == "knn":
    noise_sample_func = noise_sampler.Latent_Noise_from_kNN(train_dataset, vae, cfg.NOISE.KNN.RANK)
elif noise_name == "all_knn":
    noise_sample_func = noise_sampler.Latent_Noise_from_All_kNN(train_dataset, vae, cfg.NOISE.KNN.RANK)
else:
    raise NotImplementedError(f"{noise_name} is unknown")

################################################################################
### Configurations are re-changed to that of test.yaml.
################################################################################

cfg.merge_from_file(args.config_path)

model_name_path = model_path[:model_path.find("/model/")]

# set inference_config
shape_dict = {"latent": (1, cfg.VAE.LATENT_DIM)}
inverse_scaler = torch.nn.Identity()
sampling_fn = sampling.get_sampling_fn(cfg, sde, shape_dict, inverse_scaler, sampling_eps, noise_sampler=noise_sample_func)

if args.result_path != "":
    result_path = args.result_path
else:
    result_base_path = os.path.join(model_name_path, "result")
    os.makedirs(result_base_path, exist_ok=True)

    if "retrieve" in inference_method:
        result_dir_name = f"{inference_method}_top{cfg.RETRIEVAL.RANK}_{sde_name}_{cfg.SDE.SAMPLING.METHOD}"
    else:
        result_dir_name = f"{inference_method}_{sde_name}_{cfg.SDE.SAMPLING.METHOD}"
    
    if cfg.SDE.SAMPLING.METHOD == "pc":
        result_dir_name = f"{result_dir_name}_{cfg.SDE.SAMPLING.PREDICTOR}_{cfg.SDE.SAMPLING.CORRECTOR}"

    if args.add_name != "":
        result_dir_name = f"{result_dir_name}_{args.add_name}"

    result_path = os.path.join(result_base_path, result_dir_name)

print(f"save to {result_path}")

if os.path.exists(result_path):
    print(result_path)
    while 1:
        ans = input('The specified output dir is already exists. Overwrite? y or n: ')
        if ans == 'y':
            break
        elif ans == 'n':
            raise ValueError("Please specify correct output dir")
        else:
            print('please type y or n')
else:
    os.makedirs(result_path, exist_ok=True)

result_motion_path = os.path.join(result_path, "motion")
result_misc_path = os.path.join(result_path, "misc")
os.makedirs(result_motion_path, exist_ok=True)
os.makedirs(result_misc_path, exist_ok=True)

shutil.copy(args.config_path, result_path)

################################################################################
### Prediction
################################################################################
# TODO change here
if args.image_path == "":
    raise ValueError("TODO")
elif args.image_path == "debug":
    image, query = train_dataset[0]
    camera_intrinsic = train_dataset.info_dict["data_list"][0]["camera_intrinsic"]
else:
    image = Image.open(args.image_path)
    image = torchvision.transforms.ToTensor()(image)

    camera_intrinsic = np.load(os.path.join(os.path.dirname(args.image_path), 'camera_matrix.npy'))
image = torch.unsqueeze(image, 0)
image = image.to(device)

print("prediction start")
k = cfg.RETRIEVAL.RANK
print(f"rank: {k}")
if k < 8:
    k = 8

# get sample and score
if "retrieve" in inference_method:
    
    if "retrieve_from_motion" in inference_method:
        Retriever = Direct_Retrieval(train_dataset)
    elif "retrieve_from_MSE" in inference_method:
        Retriever = MSE_Based_Retrieval(train_dataset, model)
    elif "retrieve_from_SPE" in inference_method: # changed from retrieve_from_image to retrieve_from_SPE
        Retriever = Image_Based_Retrieval_SPE(train_dataset, model)
    elif "retrieve_from_CLIP" in inference_method:
        Retriever = CLIP_Retrieval(train_dataset)
    
    if "retrieve_from_motion" in inference_method:
        near_queries = Retriever.retrieve_k_sample(query, k=k)[0]
    else:
        near_queries, nears = Retriever.retrieve_k_sample(image)[:2]

    retrieved_query = {}
    for key in near_queries.keys():
        retrieved_query[key] = near_queries[key][:,cfg.RETRIEVAL.RANK-1].to(device)
    
    print(f"index: {nears[:,cfg.RETRIEVAL.RANK-1]}")
    retrieved_query_copy = copy.deepcopy(retrieved_query)
    if "wo_modification" in inference_method:
        pred_action = retrieved_query
    else:
        print("prediction start")
        pred_action, n = sampling_fn(model, x=retrieved_query, condition=image, N=cfg.SDE.N)
        print("done")

    pred_action = get_pos(pred_action, camera_intrinsic, (800, 1280))
    retrieved_query_copy = get_pos(retrieved_query_copy, camera_intrinsic, (800, 1280))
else:
    raise ValueError("Invalid method")
print("prediction end")

################################################################################
### Save
################################################################################

if args.image_path == "debug":
    resize = torchvision.transforms.Resize((800, 1280))
    image = resize(image)

# save retrieved data
if "retrieve" in inference_method:
    near_query_list = []
    for rank_index in range(8):
        temp = {}
        for key in near_queries.keys():
            temp[key] = near_queries[key][:,rank_index]
        near_query_list.append(temp)
    near_query_list = [get_pos(temp_query, camera_intrinsic, (800, 1280)) for temp_query in near_query_list]
    nears_img = visualize_multi_query_pos(image, near_query_list, camera_intrinsic, rot_mode=rot_mode)
    nears_img.save(os.path.join(result_motion_path, f"retrieved_motion.png"))

    vis_img = visualize_multi_query_pos(image, [retrieved_query_copy, pred_action], camera_intrinsic, rot_mode=rot_mode)
else:
    vis_img = visualize_multi_query_pos(image, [pred_action], camera_intrinsic, rot_mode=rot_mode)

vis_img.save(os.path.join(result_motion_path, f"pred_motion.png"))

# save 
for key in pred_action.keys():
    pred_action[key] = pred_action[key].cpu()
    retrieved_query_copy[key] = retrieved_query_copy[key].cpu()


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    This code is copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def output2baxter_action(query, image_size=(800, 1280), end_time=12., default_left_pose=[-96.52, 984.35, 0.4229, -0.2069,-0.5902,0.7237,-0.2917, 0.]):
    # convert uv range from [-1,1] to [0, H or W]
    uv = query["uv"]
    h, w = image_size
    u, v = uv[:,:,0], uv[:,:,1]
    u = (u + 1) / 2 * (w - 1)
    v = (v + 1) / 2 * (h - 1)
    right_uv = torch.stack([u, v], 2)

    # convert grasping value range 
    right_grasp = query["grasp_state"] * 100
    right_grasp = torch.clamp(right_grasp, min=0, max=100)

    # convert rotation 6d to quat
    right_rot = rotation_6d_to_matrix(query["rotation"])
    right_rot = R.from_matrix(right_rot[0].numpy())
    right_rot = torch.unsqueeze(torch.tensor(np.array(right_rot.as_quat())), 0)
    
    # get time_value
    _, S, _ = query["uv"].shape
    time = torch.arange(0, S) * end_time / (S-1)
    time = repeat(time, "S -> B S D", B=1, D=1)

    # concat time, default_left_pose, predicted_right_pose
    left_pose = repeat(torch.tensor(default_left_pose), "D -> B S D", B=1, S=S)
    action = torch.cat([time, left_pose, right_uv, query["z"], right_rot, right_grasp], 2)[0]
    return action.numpy()

action_array = output2baxter_action(pred_action)
retrieved_array = output2baxter_action(retrieved_query_copy)

# save_result
print("save_result to csv")    
action_csv_path = os.path.join(result_path, "pred_trajectory.csv")
retrieval_csv_path = os.path.join(result_path, "retrieved_trajectory.csv")
# head_list = ["Time", "left_u", "left_v", "left_z", "left_qx", "left_qy", "left_qz", "left_qw", "left_grip", "right_u", "right_v", "right_z", "right_qx", "right_qy", "right_qz", "right_qw", "right_grip"]
header = "Time,left_u,left_v,left_z,left_qx,left_qy,left_qz,left_qw,left_grip,right_u,right_v,right_z,right_qx,right_qy,right_qz,right_qw,right_grip"

np.savetxt(action_csv_path, action_array, delimiter=",", fmt="%.4f", header=header, comments="")
np.savetxt(retrieval_csv_path, retrieved_array, delimiter=",", fmt="%.4f", header=header, comments="")

# with open(csv_path, 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(head_list)
#     for action in action_list:
#         writer.writerow(action)