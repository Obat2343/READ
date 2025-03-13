import argparse
import sys
import os
import yaml
import time
import shutil

import torch
import wandb
import numpy as np

sys.path.append("../")

from pycode.READ.vae import VAE_Loss, Single_Class_TransformerVAE, Multi_Latent_TransformerVAE
from pycode.misc import str2bool, save_checkpoint, Time_memo, save_args, get_pos, visualize_multi_query_pos
from pycode.dataset import RLBench_DMOEBM, Baxter_Demos
from pycode.config import _C as cfg

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/Train_ACTOR.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--reset_dataset', action='store_true')
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

if args.name == "":
    if cfg.VAE.LATENT_NUM == 1:
        dir_name = f"ACTOR_frame_{args.frame}_latentdim_{cfg.VAE.LATENT_DIM}_KLD_{cfg.VAE.KLD_WEIGHT}"
    else:
        dir_name = f"ACTOR_frame_{args.frame}_latentdim_{cfg.VAE.LATENT_DIM}_latentnum_{cfg.VAE.LATENT_NUM}_KLD_{cfg.VAE.KLD_WEIGHT}"
else:
    dir_name = args.name

if args.add_name != "":
    dir_name = f"{dir_name}_{args.add_name}"

device = args.device
rot_mode = "6d"
max_iter = cfg.OUTPUT.MAX_ITER
save_iter = cfg.OUTPUT.SAVE_ITER
eval_iter = cfg.OUTPUT.EVAL_ITER
log_iter = cfg.OUTPUT.LOG_ITER
batch_size = cfg.DATASET.BATCH_SIZE
input_keys = ["uv","z","rotation","grasp_state"]
input_dims = [2, 1, 6, 1]

save_dir = os.path.join(cfg.OUTPUT.BASE_DIR, cfg.DATASET.NAME)
save_path = os.path.join(save_dir, dir_name)
print(f"save path:{save_path}")
# os.makedirs(save_path, exist_ok=True)
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

if not args.debug:
    wandb.login()
    run = wandb.init(project='ACTOR', 
                    config=obj, save_code=True, group=cfg.DATASET.RLBENCH.TASK_NAME, name=dir_name, dir=save_dir)

model_save_dir = os.path.join(save_path, "model")
log_dir = os.path.join(save_path, 'log')
vis_dir = os.path.join(save_path, 'vis')
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# copy config file
shutil.copy(sys.argv[0], save_path)
if len(args.config_file) > 0:
    shutil.copy(os.path.abspath(args.config_file), save_path)

# save args
argsfile_path = os.path.join(save_path, "args.json")
save_args(args,argsfile_path)

# set dataset
if cfg.DATASET.NAME == "RLBench":
    dataclass = RLBench_DMOEBM
elif cfg.DATASET.NAME == "Baxter":
    dataclass = Baxter_Demos
else:
    raise ValueError("Invalid dataset name")

train_dataset  = dataclass("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode, keys=input_keys)
train_dataset.without_image = True
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dataset = dataclass("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode, keys=input_keys)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)
    
# set model
input_size = 10 * (args.frame + 1)
model = Single_Class_TransformerVAE(input_keys, input_dims, args.frame + 1, latent_dim=cfg.VAE.LATENT_DIM, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"]).to(device)
# model = Multi_Latent_TransformerVAE(input_keys, input_dims, args.frame + 1, latent_dim=cfg.VAE.LATENT_DIM, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"]).to(device)

# temp_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers=2, shuffle=False)
vae_loss = VAE_Loss(rot_mode=rot_mode, kld_weight=cfg.VAE.KLD_WEIGHT)

loss_hist = np.array([])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)

iteration = 0
flag = False
start = time.time()
time_memo = Time_memo()
for epoch in range(100000):
    for data in train_dataloader:
        _, h_query = data
        for key in h_query.keys():
            h_query[key] = h_query[key].to(device)
            
        optimizer.zero_grad()
        pred_dict, z, mu, log_var = model(h_query)
        
        loss, loss_dict = vae_loss(pred_dict, h_query, mu, log_var)
        
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if iteration % log_iter == 0:
            end = time.time()
            cost = (end - start) / (iteration+1)
            print(f"train iter: {iteration} cost: {cost:.4g} loss: {loss.item():.4g} uv: {loss_dict['train/uv_loss']:.4g} KLD: {loss_dict['train/KLD']:.4g}")
            
            if not args.debug:
                wandb.log(loss_dict, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        if iteration % eval_iter == 0:
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    image, h_query  = data
                    for key in h_query.keys():
                        h_query[key] = h_query[key].to(device)
                    pred_dict, z, mu, log_var = model(h_query)
                    loss, loss_dict = vae_loss(pred_dict, h_query, mu, log_var, mode="val")
                    print(f"val iter: {iteration} loss: {loss.item():.4g} uv: {loss_dict['val/uv_loss']:.4g} KLD: {loss_dict['val/KLD']:.4g}")

            for key in pred_dict.keys():
                pred_dict[key] = pred_dict[key].cpu()
                h_query[key] = h_query[key].cpu()

            h_query = get_pos(h_query, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
            pred_dict = get_pos(pred_dict, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
            pil_img = visualize_multi_query_pos(image, [pred_dict, h_query], train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
            pil_img.save(os.path.join(vis_dir, f"pos_img_val_{iteration}.png"))

            if not args.debug:
                wandb.log(loss_dict, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        if iteration % save_iter == 0:
            # save model
            model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

        if iteration == max_iter + 1:
            flag = True
            if not args.debug:
                wandb.finish()
            break
            # sys.exit()

        iteration += 1

    if flag == True:
        break