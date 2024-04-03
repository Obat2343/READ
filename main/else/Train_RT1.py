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
from robotic_transformer_pytorch import MaxViT, RT1
from einops import rearrange, repeat

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/RLBench_RT1.yaml', metavar='FILE', help='path to config file')
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
    dir_name = f"RT1_frame_{args.frame}_mode_{args.rot_mode}_nbin_256_ver2"
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
vit = MaxViT(
    num_classes = 1000, # not necessary
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 32,
    depth = (2, 2, 5, 2),
    window_size = 8, # important when you changed the image size from 224
    mbconv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)

vit.conv_stem = torch.nn.Sequential(
                    torch.nn.Conv2d(4, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                    torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                )

model = RT1(
    vit = vit,
    num_actions = 10 * 101,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2
)

model = model.to(device)

string = cfg.DATASET.RLBENCH.TASK_NAME

# 大文字を探して、小文字に変換する
instruction = ""
for i, char in enumerate(string):
    if char.isupper():
        # 文頭でない場合は、前に空白を挿入する
        if i > 0:
            instruction += " "
        instruction += char.lower()
    else:
        instruction += char

print(instruction)
instruction = [instruction]

# set optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)
Train_Loss = torch.nn.CrossEntropyLoss()
# Eval_loss = Diffusion_Loss()

class query_bin_converter():
    
    def __init__(self, target_key=['rotation', 'grasp_state', 'uv', 'z'], target_dim=[6,1,2,1], n_bin=256, device="cuda"):
        self.target_key = target_key
        self.target_dim = target_dim
        self.n_bin = n_bin
        
        boundaries = torch.arange(0, self.n_bin) / (self.n_bin-1)
        boundaries = boundaries.to(device)
        self.boundaries = (boundaries - 0.5) * 2
        self.boundaries_z = boundaries + 1
        
    def get_onehot_index_query(self, query):
        onehot_index_query = {}
        for key in self.target_key:
            if key == "z":
                onehot_index_query[key] = torch.clamp(torch.bucketize(query[key], self.boundaries_z), 0, self.n_bin - 1)
            else:
                onehot_index_query[key] = torch.clamp(torch.bucketize(query[key], self.boundaries), 0, self.n_bin - 1)
            
        return onehot_index_query
    
    def get_onehot_index_vec(self, query):
        onehot_index_query = {}
        for key in self.target_key:
            if key == "z":
                onehot_index_query[key] = torch.bucketize(query[key], self.boundaries_z)
            else:
                onehot_index_query[key] = torch.bucketize(query[key], self.boundaries) # B S D
        
        onehot_index_vec = torch.cat([onehot_index_query[key] for key in self.target_key], 2) # B S (D1, D2, D3,...
        onehot_index_vec = rearrange(onehot_index_vec, "B S D -> B (S D)")
        onehot_index_vec = torch.clamp(onehot_index_vec, 0, self.n_bin - 1)

        return onehot_index_vec
    
    def onehot_index_vec2query(self, onehot_index_vec):
        onehot_index_vec = rearrange(onehot_index_vec, "B (S D) -> B S D", D=sum(self.target_dim))
        
        query = {}
        sum_dim = 0
        for key, dim in zip(self.target_key, self.target_dim):
            index_query = onehot_index_vec[:,:,sum_dim:sum_dim+dim]
            
            B, S, D = index_query.shape
            index_query = rearrange(index_query, "B S D -> (B S D)")
            if key == "z":
                value = torch.gather(self.boundaries_z, 0, index_query)
            else:
                value = torch.gather(self.boundaries, 0, index_query)
            value = rearrange(value, "(B S D) -> B S D", B=B, S=S, D=D)
            query[key] = value
            
            sum_dim += dim
            
        return query

converter = query_bin_converter(device=device)

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

        optimizer.zero_grad()
        video = torch.unsqueeze(image, 2)
        pred_logits = torch.squeeze(model(video, instruction),1)

        target = converter.get_onehot_index_vec(query)
        loss = Train_Loss(pred_logits.view(-1, 256), target.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log
        if iteration % log_iter == 0:
            end = time.time()
            cost = (end - start) / (iteration+1)
            print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss.item():.4g}')
            
            if args.log2wandb:
                wandb.log({"train/loss": loss.item()}, step=iteration)
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
                    model.eval()
                    image, query = data
                    image = image.to(device)
                    for key in query.keys():
                        query[key] = query[key].to(device)

                    optimizer.zero_grad()
                    video = torch.unsqueeze(image, 2)
                    pred_logits = torch.squeeze(model(video, instruction),1)

                    target = converter.get_onehot_index_vec(query)
                    loss = Train_Loss(pred_logits.view(-1, 256), target.view(-1))
                    print(f'Val Iter: {iteration} Loss: {loss.item():.4g}')
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

                    model.train()
            
                if args.log2wandb:
                    wandb.log({"val/loss": loss.item()}, step=iteration)

        # save model
        if iteration % save_iter == 0:
            model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
            save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

        if iteration == max_iter + 1:
            sys.exit()
        
        iteration += 1