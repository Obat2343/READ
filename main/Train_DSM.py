import argparse
import sys
import os
import yaml
import time
import shutil

import torch
import wandb

sys.path.append("../")
from pycode.config import _C as cfg
from pycode.dataset import RLBench_DMOEBM
from pycode.misc import str2bool, save_checkpoint, get_pos, visualize_multi_query_pos, convert_rotation_6d_to_matrix, save_args, load_checkpoint
from pycode.model.diffusion import Denoising_Score_Matching, DSM_Loss
from pycode.model.Motion_Gen import Single_Class_TransformerVAE

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/RLBench_DSM.yaml', metavar='FILE', help='path to config file')
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

if args.name == "":
    if cfg.DSM.NOISE.TYPE == "gaussian":
        dir_name = f"DSM_normal_frame_{args.frame}_mode_{args.rot_mode}_step_{cfg.DSM.STEP}_start_{cfg.DSM.NOISE.GAUSSIAN.START_STD}_end_{cfg.DSM.NOISE.GAUSSIAN.END_STD}"
    elif cfg.DSM.NOISE.TYPE == "latent_gaussian":
        dir_name = f"DSM_Latent_frame_{args.frame}_mode_{args.rot_mode}_step_{cfg.DSM.STEP}_start_{cfg.DSM.NOISE.GAUSSIAN.START_STD}_end_{cfg.DSM.NOISE.GAUSSIAN.END_STD}"
    else:
        print(cfg.DSM.NOISE.TYPE)
        raise ValueError("invalid noise type")
else:
    dir_name = args.name

if args.add_name != "":
    dir_name = f"{dir_name}_{args.add_name}"

device = args.device
rot_mode = args.rot_mode
max_iter = cfg.OUTPUT.MAX_ITER
save_iter = cfg.OUTPUT.SAVE_ITER
eval_iter = cfg.OUTPUT.EVAL_ITER
log_iter = cfg.OUTPUT.LOG_ITER
batch_size = cfg.DATASET.BATCH_SIZE

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
train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dataset = RLBench_DMOEBM("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
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
atten_dropout_rate = cfg.MODEL.ATTEN_DROPOUT_RATE
query_emb_dim = cfg.MODEL.QUERY_EMB_DIM
num_atten_block = cfg.MODEL.NUM_ATTEN_BLOCK

max_steps = cfg.DSM.STEP

if cfg.DSM.NOISE.TYPE == "latent_gaussian":
    # load VAE
    VAE_path = f"../global_result/RLBench/{cfg.DATASET.RLBENCH.TASK_NAME}/Transformer_VAE_frame_{args.frame}_latentdim_64_mode_{rot_mode}_KLD_0/model/model_iter100000.pth"
    VAE = Single_Class_TransformerVAE(["uv","z","rotation","grasp_state"],[2,1,rot_dim,1], args.frame+1, latent_dim=64, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
    VAE, _, _, _, _ = load_checkpoint(VAE, VAE_path)
    VAE.eval()
    VAE.to(device)
else:
    VAE = "none"

model = Denoising_Score_Matching(["uv","z","rotation","grasp_state"], [2,1,rot_dim,1], cfg.DSM, VAE=VAE,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
model = model.to(device)

# set optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)
Train_loss = DSM_Loss()

##### start training #####
iteration = 0
start = time.time()
for epoch in range(10000000):
    for data in train_dataloader:
        image, h_query = data
        image = image.to(device)
        for key in h_query.keys():
            h_query[key] = h_query[key].to(device)
        B,_,_,_ = image.shape
        t = torch.randint(1, max_steps+1, (B,), device=device).long()

        optimizer.zero_grad()
        pred_dict, noise, stds, info = model(image, h_query, t)

        loss, loss_dict = Train_loss(pred_dict, noise, stds)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log
        if iteration % log_iter == 0:
            end = time.time()
            cost = (end - start) / (iteration+1)
            print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss_dict["train/loss"]:.4g} uv:{loss_dict["train/uv"]:.4g}, z:{loss_dict["train/z"]:.4g}, rot:{loss_dict["train/rotation"]:.4g}, grasp:{loss_dict["train/grasp_state"]:.4g}')
            
            if args.log2wandb:
                wandb.log(loss_dict, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        # evaluate model
        if iteration % eval_iter == 0:
            with torch.no_grad():

                for data in val_dataloader:
                    model.eval()
                    image, h_query = data
                    image = image.to(device)
                    for key in h_query.keys():
                        h_query[key] = h_query[key].to(device)
                    t = torch.randint(1, max_steps+1, (100,), device=device).long()

                    optimizer.zero_grad()
                    pred_dict, noise, stds, info = model(image, h_query, t)
                    loss, loss_dict = Train_loss(pred_dict, noise, stds, mode="val")
                    print(f'Val Iter: {iteration} Loss: {loss_dict["val/loss"]:.4g} uv:{loss_dict["val/uv"]:.4g}, z:{loss_dict["val/z"]:.4g}, rot:{loss_dict["val/rotation"]:.4g}, grasp:{loss_dict["val/grasp_state"]:.4g}')

                    model.train()
            
                if args.log2wandb:
                    wandb.log(loss_dict, step=iteration)

        # save model
        if iteration % save_iter == 0:
            model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

        if iteration == max_iter + 1:
            sys.exit()
        
        iteration += 1