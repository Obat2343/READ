import argparse
import sys
import os
import yaml
import time
import shutil
import json

import torch
import wandb

sys.path.append("../")
from pycode.config import _C as cfg
from pycode.dataset import RLBench_DMOEBM
from pycode.misc import str2bool, save_checkpoint, get_pos, visualize_multi_query_pos, convert_rotation_6d_to_matrix, save_args
from pycode.DF.model import SPE_Energy_Predictor
from pycode.DF import losses
from pycode.SDE import sde_lib

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/Train_PseudoEnergy.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--reset_dataset', type=str2bool, default=False)
parser.add_argument('--frame', type=int, default=100)
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
rot_mode = "6d"
max_iter = cfg.OUTPUT.MAX_ITER
save_iter = cfg.OUTPUT.SAVE_ITER
eval_iter = cfg.OUTPUT.EVAL_ITER
log_iter = cfg.OUTPUT.LOG_ITER
batch_size = cfg.DATASET.BATCH_SIZE
input_keys = ["uv","z","rotation","grasp_state"]
input_dims = [2, 1, 6, 1]

if args.name == "":
    if cfg.SDE.TRAINING.CONTINUOUS == True:
        dir_name = f"PseudoEnergy_{cfg.SDE.NAME}_frame_{args.frame}"
    else:
        dir_name = f"PseudoEnergy_{cfg.SDE.NAME}_frame_{args.frame}_step_{cfg.SDE.N}"
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
    run = wandb.init(project='{}-{}'.format(cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_NAME), entity='tendon',
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
train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode, keys=input_keys)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dataset = RLBench_DMOEBM("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode, keys=input_keys)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)

# set model
conv_dims = cfg.MODEL.CONV_DIMS
enc_depths = cfg.MODEL.ENC_DEPTHS
enc_layers = cfg.MODEL.ENC_LAYERS

dec_depths = cfg.MODEL.DEC_DEPTHS
dec_layers = cfg.MODEL.DEC_LAYERS

extractor_name = cfg.MODEL.EXTRACTOR_NAME
predictor_name = cfg.MODEL.PREDICTOR_NAME

conv_droppath_rate = cfg.MODEL.CONV_DROP_PATH_RATE
query_emb_dim = cfg.MODEL.QUERY_EMB_DIM

img_guidance_rate = cfg.DIFFUSION.IMG_GUIDE
max_steps = cfg.DIFFUSION.STEP

model = SPE_Energy_Predictor(input_keys, input_dims, dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate, img_guidance_rate=img_guidance_rate)
model = model.to(device)

# set sde
sde = sde_lib.VPSDE(keys=input_keys, beta_min=cfg.SDE.VPSDES.BETA_MIN, beta_max=cfg.SDE.VPSDES.BETA_MAX, N=cfg.SDE.N)
# if args.log2wandb == False:
#     print("sqrt_alphas_cumprod")
#     print(sde.sqrt_alphas_cumprod)
#     print("sqrt_1m_alphas_cumprod")
#     print(sde.sqrt_1m_alphas_cumprod)

# set optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)
Train_loss = losses.get_loss_fn(sde, energy=True, train=True, reduce_mean=True, continuous=cfg.SDE.TRAINING.CONTINUOUS,
                                    likelihood_weighting=False)
Val_loss = losses.get_loss_fn(sde, energy=True, train=False, reduce_mean=True, continuous=cfg.SDE.TRAINING.CONTINUOUS,
                                    likelihood_weighting=False)

##### start training #####
iteration = 0
start = time.time()
for epoch in range(10000000):
    for data in train_dataloader:
        image, query = data
        image = image.to(device)
        for key in query.keys():
            query[key] = query[key].to(device)

        # optimization
        optimizer.zero_grad()
        loss, loss_dict = Train_loss(model, query, condition=image)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log
        if iteration % log_iter == 0:
            end = time.time()
            cost = (end - start) / (iteration+1)
            print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss_dict["train/loss"]:.4g}')
            
            if args.log2wandb:
                wandb.log(loss_dict, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        # save model
        if iteration % save_iter == 0:
            model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

        if iteration == max_iter + 1:
            sys.exit()
        
        iteration += 1