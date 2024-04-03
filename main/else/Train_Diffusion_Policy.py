import argparse
import sys
import os
import yaml
import time
import shutil
import json

import torch
import wandb
import torch.nn as nn

sys.path.append("../")
sys.path.append("../git/diffusion_policy")
from pycode.config import _C as cfg
from pycode.dataset import RLBench_DMOEBM
from pycode.misc import str2bool, save_checkpoint, save_args, Concat_query_keys
# from pycode.model.diffusion import Denoising_Diffusion, Diffusion_Loss, calculate_end, calculate_end_for_cosine

from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import replace_submodules

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/RLBench_DiffusionPolicy.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--reset_dataset', type=str2bool, default=False)
parser.add_argument('--frame', type=int, default=100)
parser.add_argument('--rot_mode', type=str, default="6d")
args = parser.parse_args()

##### config #####
# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

    # set config_file to wandb
    with open(args.config_file) as file:
        obj = yaml.safe_load(file)

device = args.device
rot_mode = args.rot_mode
max_iter = cfg.OUTPUT.MAX_ITER
save_iter = cfg.OUTPUT.SAVE_ITER
eval_iter = cfg.OUTPUT.EVAL_ITER
log_iter = cfg.OUTPUT.LOG_ITER
batch_size = cfg.DATASET.BATCH_SIZE

if args.name == "":
    dir_name = f"Diffusion_Policy_frame_{args.frame}_mode_{args.rot_mode}_step_{cfg.DIFFUSION.STEP}_linear"
else:
    dir_name = args.name

if args.add_name != "":
    dir_name = f"{dir_name}_{args.add_name}"

save_dir = os.path.join(cfg.OUTPUT.BASE_DIR, cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_NAME)
save_path = os.path.join(save_dir, dir_name)
print(f"save path:{save_path}")
if os.path.exists(save_path):
    while 1:
        ans = input('The specified output dir is already exists. Overwrite? y or n: ')
        if ans == 'y':
            break
        elif ans == 'n':
            raise ValueError("Please specify correct output dir")
        else:
            print('please type y or n')
else:
    os.makedirs(save_path)

if rot_mode == "quat":
    rot_dim = 4
elif rot_mode == "6d":
    rot_dim = 6
else:
    raise ValueError("TODO")

if args.log2wandb:
    wandb.login()
    run = wandb.init(project='{}-{}'.format(cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_NAME), 
                    config=obj, save_code=True, name=dir_name, dir=save_dir)

model_save_dir = os.path.join(save_path, "model")
log_dir = os.path.join(save_path, 'log')
vis_dir = os.path.join(save_path, 'vis')
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# copy source code
shutil.copy(sys.argv[0], save_path)
if len(args.config_file) > 0:
    shutil.copy(os.path.abspath(args.config_file), save_path)

# save args
argsfile_path = os.path.join(save_path, "args.json")
save_args(args,argsfile_path)

# set dataset
train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dataset = RLBench_DMOEBM("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)

# set model
max_steps = cfg.DIFFUSION.STEP
pred_horizon = args.frame
obs_horizon = 1
# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 10
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 10

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
vision_encoder = get_resnet('resnet18')
vision_encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = replace_submodules(
                    root_module=vision_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features//16, 
                            num_channels=x.num_features)
                    )

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

# for this demo, we use DDPMScheduler with 100 diffusion iterations
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_steps,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='linear',
    beta_start=1e-5,
    beta_end=0.02,
    # clip output to [-1,1] to improve stability
    clip_sample=False,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    model=nets,
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(), 
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=max_iter
)

converter = Concat_query_keys()
##### start training #####
iteration = 0
start = time.time()
for epoch in range(10000000):
    for data in train_dataloader:
        image, query = data
        image = image.to(device)
        for key in query.keys():
            query[key] = query[key].to(device)
        B,_,_,_ = image.shape
        
        action = converter.concat_query(query)        
        nimage = torch.unsqueeze(image, 1).to(device)
        nagent_pos = action[:,:obs_horizon].to(device)
        naction = action[:,obs_horizon:].to(device)
        B = nagent_pos.shape[0]

        # encoder vision features
        image_features = nets['vision_encoder'](
            nimage.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage.shape[:2],-1)
        # (B,obs_horizon,D)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, 
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(
            naction, noise, timesteps)
        
        # predict the noise residual
        noise_pred = nets['noise_pred_net'](
            noisy_actions, timesteps, global_cond=obs_cond)
        
        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(noise_pred_net)

        # log
        if iteration % log_iter == 0:
            end = time.time()
            cost = (end - start) / (iteration+1)
            print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss.item():.4g}')
            
            if args.log2wandb:
                wandb.log({"train/loss":loss.item()}, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        # evaluate model
        if iteration % eval_iter == 0:
            with torch.no_grad():
                # pred_dict = get_pos(pred_dict, train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
                # if rot_mode == "6d":
                #     h_query, noise_query, pred_dict = convert_rotation_6d_to_matrix([h_query, noise_query, pred_dict])
                # loss_dict = Eval_loss(pred_dict, h_query)
                # print(f'Train: {iteration} Pos:{loss_dict["train/pos_error"]:.4g}, z:{loss_dict["train/z_error"]:.4g}, rot:{loss_dict["train/rot_error"]:.4g}, grasp:{loss_dict["train/grasp_accuracy"]:.4g}')
                
                # for key in h_query.keys():
                #     h_query[key] = h_query[key].cpu()
                #     noise_query[key] = noise_query[key].cpu()
                #     try:
                #         pred_dict[key] = pred_dict[key].cpu()
                #     except:
                #         pass

                # vis_img = visualize_multi_query_pos(image, [h_query, noise_query, pred_dict], train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
                # vis_img.save(os.path.join(vis_dir, f"pos_img_train_{iteration}.png"))
                
                # if args.log2wandb:
                    # wandb.log(loss_dict, step=iteration)

                for data in val_dataloader:
                    image, query = data
                    image = image.to(device)
                    for key in query.keys():
                        query[key] = query[key].to(device)

                    action = converter.concat_query(query)        
                    nimage = torch.unsqueeze(image, 1).to(device)
                    nagent_pos = action[:,:obs_horizon].to(device)
                    naction = action[:,obs_horizon:].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, 
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    
                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)
                    
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    print(f'Val Iter: {iteration} Loss: {loss.item():.4g}')
                    optimizer.zero_grad()

                    # pred_dict = get_pos(pred_dict, train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
                    
                    # if rot_mode == "6d":
                    #     h_query, noise_query, pred_dict = convert_rotation_6d_to_matrix([h_query, noise_query, pred_dict])
                    # loss_dict = Eval_loss(pred_dict, h_query, mode="val")
                    # print(f'Val: {iteration} Pos:{loss_dict["val/pos_error"]:.4g}, z:{loss_dict["val/z_error"]:.4g}, rot:{loss_dict["val/rot_error"]:.4g}, grasp:{loss_dict["val/grasp_accuracy"]:.4g}')
                    
                    # for key in pred_dict.keys():
                    #     h_query[key] = h_query[key].cpu()
                    #     noise_query[key] = noise_query[key].cpu()
                    #     pred_dict[key] = pred_dict[key].cpu()

                    # vis_img = visualize_multi_query_pos(image, [h_query, noise_query, pred_dict], train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
                    # vis_img.save(os.path.join(vis_dir, f"pos_img_val_{iteration}.png"))
            
                if args.log2wandb:
                    wandb.log({"val/loss":loss.item()}, step=iteration)

        # save model
        if iteration % save_iter == 0:
            ema_net = ema.averaged_model
            model_save_path = os.path.join(model_save_dir, f"ema_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(ema_net, optimizer, epoch, iteration, model_save_path)

            model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(nets, optimizer, epoch, iteration, model_save_path)

        if iteration == max_iter + 1:
            sys.exit()
        
        iteration += 1