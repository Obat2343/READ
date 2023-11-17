import argparse
import sys
import os
import yaml
import time
import shutil
import random

import torch
import wandb

sys.path.append("../")
from pycode.config import _C as cfg
from pycode.dataset import RLBench_Retrieval
from pycode.misc import str2bool, save_checkpoint, save_args, load_checkpoint
from pycode.DF.model import SPE_Continuous_Latent_Diffusion, Timm_Continuous_Latent_Diffusion, AvgPool_Continuous_Latent_Diffusion, ConvPool_Continuous_Latent_Diffusion
from pycode.DF.vae import Single_Class_TransformerVAE
from pycode.DF import losses, sde_lib, noise_sampler

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/Train_LatentContinuousDiff.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--vae_path', type=str, default="")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--frame', type=int, default=100)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--reset_dataset', action='store_true')
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
    assert cfg.SDE.TRAINING.CONTINUOUS == True
    if cfg.NOISE.NAME == "gaussian":
        dir_name = f"Latent_ContinuousDiff_{cfg.SDE.NAME}_rank_{cfg.RETRIEVAL.RANK}_ModelIO_{cfg.MODEL.INOUT}_VAEdim_{cfg.VAE.LATENT_DIM}_frame_{args.frame}"
    else:
        dir_name = f"Latent_ContinuousDiff_{cfg.SDE.NAME}_{cfg.NOISE.NAME}_rank_{cfg.RETRIEVAL.RANK}_ModelIO_{cfg.MODEL.INOUT}_VAEdim_{cfg.VAE.LATENT_DIM}_frame_{args.frame}"
    
    if cfg.MODEL.NAME != "Convnext-UNet":
        dir_name = f"{dir_name}_{cfg.MODEL.NAME}"
    
    if cfg.MODEL.TARGET != "auto":
        dir_name = f"{dir_name}_{cfg.MODEL.TARGET}"
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

if not args.debug:
    wandb.login()
    run = wandb.init(project='Latent_Diffusion', entity='tendon', group=cfg.DATASET.RLBENCH.TASK_NAME,
                    config=obj, save_code=True, name=dir_name, dir=save_dir)

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
train_dataset  = RLBench_Retrieval("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode, keys=input_keys, rank=cfg.RETRIEVAL.RANK)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dataset = RLBench_Retrieval("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode, keys=input_keys, rank=cfg.RETRIEVAL.RANK)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)

# set vae
if args.vae_path == "":
    vae_path = f"../weights/{cfg.DATASET.NAME}/{cfg.DATASET.RLBENCH.TASK_NAME}/ACTOR_frame_{args.frame}_latentdim_{cfg.VAE.LATENT_DIM}_KLD_{cfg.VAE.KLD_WEIGHT}/model/model_iter50000.pth"
else:
    vae_path = args.vae_path

vae = Single_Class_TransformerVAE(input_keys, input_dims, args.frame + 1, latent_dim=cfg.VAE.LATENT_DIM).to(device)
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
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
elif model_name == "Convnext-UNet-avgpool":
    model = AvgPool_Continuous_Latent_Diffusion(input_keys, input_dims, vae, vae_latent_dim, inout_type,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
elif model_name == "Convnext-UNet-convpool":
    model = ConvPool_Continuous_Latent_Diffusion(input_keys, input_dims, vae, vae_latent_dim, inout_type,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
else:
    model = Timm_Continuous_Latent_Diffusion(input_keys, input_dims, model_name, vae, vae_latent_dim, inout_type, img_size=256, input_dim=4, pretrained=True,
                    query_emb_dim=query_emb_dim)
model = model.to(device)

# set sde
sde_name = cfg.SDE.NAME
assert sde_name == 'irsde'
sde = sde_lib.IRSDE(keys=input_keys, lamda=cfg.SDE.IRSDES.LAMDA, theta_1=cfg.SDE.IRSDES.THETA, schedule="linear")

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

# set optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)
Train_loss = losses.get_loss_fn(sde, target=cfg.MODEL.TARGET, noise_sampler=noise_sample_func, energy=False, train=True, reduce_mean=True, continuous=cfg.SDE.TRAINING.CONTINUOUS,
                                    likelihood_weighting=False, model_type=cfg.MODEL.INOUT)
Val_loss = losses.get_loss_fn(sde, target=cfg.MODEL.TARGET, noise_sampler=noise_sample_func,energy=False, train=False, reduce_mean=True, continuous=cfg.SDE.TRAINING.CONTINUOUS,
                                    likelihood_weighting=False, model_type=cfg.MODEL.INOUT)

##### start training #####
iteration = 0
start = time.time()
for epoch in range(10000000):
    for data in train_dataloader:
        image, query, pos_image, pos_query, neg_image, neg_query = data
        image, pos_image = image.to(device), pos_image.to(device)
        for key in query.keys():
            query[key] = query[key].to(device)
            pos_query[key] = pos_query[key].to(device)
        
        # optimization
        optimizer.zero_grad()
        loss, loss_dict = Train_loss(model, query, condition=image, retrieved=pos_query)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log
        if iteration % log_iter == 0:
            end = time.time()
            cost = (end - start) / (iteration+1)
            print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss_dict["train/loss"]:.4g}')
            
            if not args.debug:
                wandb.log(loss_dict, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        if iteration % eval_iter == 0:
            for data in val_dataloader:
                image, query, _, _, _, _ = data
                image = image.to(device)
                for key in query.keys():
                    query[key] = query[key].to(device)
                break
                
            # retrieval
            index = random.randint(1, cfg.RETRIEVAL.RANK)
            _, retrieved_query = train_dataset.retrieve_k_sample(query, k=index)
            for key in retrieved_query.keys():
                retrieved_query[key] = retrieved_query[key].to(device)

            with torch.no_grad():
                loss, loss_dict = Val_loss(model, query, condition=image, retrieved=retrieved_query)
            
            print(f'Val Iter: {iteration} Loss: {loss_dict["val/loss"]:.4g}')

            if not args.debug:
                wandb.log(loss_dict, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

        # save model
        if iteration % save_iter == 0:
            model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

        if iteration == max_iter + 1:
            sys.exit()
        
        iteration += 1