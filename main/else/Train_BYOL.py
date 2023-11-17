import argparse
import sys
import os
import yaml
import time
import shutil
import datetime
from typing import Dict, Callable, List

import torch
import torch.nn as nn
import wandb
import numpy as np
from torchvision import models
from byol_pytorch import BYOL
from torchvision import transforms as T

sys.path.append("../")

from pycode.misc import str2bool, save_checkpoint, Time_memo, save_args, visualize_multi_query_pos
from pycode.dataset import RLBench_DMOEBM
from pycode.config import _C as cfg

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/RLBench_BYOL.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--reset_dataset', type=str2bool, default=False)
parser.add_argument('--frame', type=int, default=100)
parser.add_argument('--rot_mode', type=str, default="6d")
parser.add_argument('--without_crop', type=str, default=False)
parser.add_argument('--group_norm', type=str2bool, default=False)
parser.add_argument('--tasks', nargs="*", type=str) # PutRubbishInBin StackWine CloseBox PushButton ReachTarget TakePlateOffColoredDishRack PutKnifeOnChoppingBoard StackBlocks
args = parser.parse_args()

##### config #####
# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

    # set config_file to wandb
    with open(args.config_file) as file:
        obj = yaml.safe_load(file)


task_list = args.tasks

base_yamlname = os.path.basename(args.config_file)
head, ext = os.path.splitext(args.config_file)
dt_now = datetime.datetime.now()
temp_yaml_path = f"{head}_{dt_now.year}{dt_now.month}{dt_now.day}_{dt_now.hour}:{dt_now.minute}:{dt_now.second}{ext}"
shutil.copy(os.path.abspath(args.config_file), temp_yaml_path)

for task_name in task_list:
    cfg.DATASET.RLBENCH.TASK_NAME = task_name

    if args.name == "":
        dir_name = f"BYOL_wo_crop_{args.without_crop}"
    else:
        dir_name = args.name

    if args.group_norm:
        dir_name = f"{dir_name}_gn"

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
        shutil.copy(temp_yaml_path, os.path.join(save_path, base_yamlname))

    # save args
    argsfile_path = os.path.join(save_path, "args.json")
    save_args(args,argsfile_path)

    # set dataset
    train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = RLBench_DMOEBM("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)
        
    # set model
    input_size = (2 + 1 + rot_dim + 1) * (args.frame + 1)
    model = models.resnet50(pretrained=True).to(device)

    def replace_submodules(
            root_module: nn.Module, 
            predicate: Callable[[nn.Module], bool], 
            func: Callable[[nn.Module], nn.Module]) -> nn.Module:
        """
        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
        """
        if predicate(root_module):
            return func(root_module)

        bn_list = [k.split('.') for k, m 
            in root_module.named_modules(remove_duplicate=True) 
            if predicate(m)]
        for *parent, k in bn_list:
            parent_module = root_module
            if len(parent) > 0:
                parent_module = root_module.get_submodule('.'.join(parent))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all BN are replaced
        bn_list = [k.split('.') for k, m 
            in root_module.named_modules(remove_duplicate=True) 
            if predicate(m)]
        assert len(bn_list) == 0
        return root_module

    if args.group_norm:
        model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

    if args.without_crop:
        customAug = T.Compose([T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])
    else:
        customAug = T.Compose([T.RandomResizedCrop(256, scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])

    learner = BYOL(
            model,
            image_size = 256,
            hidden_layer = 'avgpool',
            augment_fn = customAug
        )

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

    iteration = 0
    flag = False
    start = time.time()
    time_memo = Time_memo()
    for epoch in range(100000):
        for data in train_dataloader:
            img, _ = data
            rgb = img[:,:3]
            loss = learner(rgb.to(device))

            optimizer.zero_grad()
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                learner.update_moving_average()
            
            if args.log2wandb:
                wandb.log({"train/loss":loss.item()}, step=iteration)
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

            if iteration % log_iter == 0:
                end = time.time()
                cost = (end - start) / (iteration+1)
                print(f"train iter: {iteration} cost: {cost:.4g} loss: {loss.item():.4g}")

            if iteration % eval_iter == 0:
                model.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        image, _  = data
                        rgb = img[:,:3]
                        loss = learner(rgb.to(device))
                        print(f"val iter: {iteration} loss: {loss.item():.4g}")

                    if args.log2wandb:
                        wandb.log({"val/loss":loss.item()}, step=iteration)

            if iteration % save_iter == 0:
                # save model
                model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
                save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

            if iteration == max_iter + 1:
                flag = True
                break
                # sys.exit()

            iteration += 1

        if flag == True:
            wandb.finish()
            break

os.remove(temp_yaml_path)