import rlbench
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

import os
import csv
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

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tasks', nargs="*", type=str, required=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--off_screen', action='store_true')
parser.add_argument('--eval_dataset', type=str, default='RLBench4')
parser.add_argument('--eval_index',type=int, default=0)
parser.add_argument('--motion_index',type=int, default=1)
parser.add_argument('--config_path', type=str, default="../config/Test_ContinuousDiff.yaml")

args = parser.parse_args()

### SET SIM ###

# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'temp'

obs_config = ObservationConfig()
obs_config.set_all(True)

# change action mode
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(),
    gripper_action_mode=Discrete()
)

# set up enviroment
print(f"off screen: {args.off_screen}")
env = Environment(
    action_mode, DATASET, obs_config, args.off_screen)

env.launch()
env._scene._cam_front.set_resolution([256,256])
env._scene._cam_front.set_position(env._scene._cam_front.get_position() + np.array([0.3,0,0.3]))

env._scene._cam_over_shoulder_left.set_resolution([256,256])
env._scene._cam_over_shoulder_left.set_position(np.array([0.32500029, 1.54999971, 1.97999907]))
env._scene._cam_over_shoulder_left.set_orientation(np.array([ 2.1415925 ,  0., 0.]))

env._scene._cam_over_shoulder_right.set_resolution([256,256])
env._scene._cam_over_shoulder_right.set_position(np.array([0.32500029, -1.54999971, 1.97999907]))
env._scene._cam_over_shoulder_right.set_orientation(np.array([-2.1415925,  0., math.pi]))

import sys
sys.path.append("../")

import torch
import torchvision

from pycode.dataset import RLBench_DMOEBM
from pycode.config import _C as cfg
from pycode.misc import load_checkpoint, convert_rotation_6d_to_matrix, visualize_inf_query, get_pos, visualize_multi_query_pos
from pycode.misc import calculate_euclid_pos, calculate_euclid_angle, calculate_euclid_grasp, output2action, check_img, get_gt_pose, make_video, get_concat_h

### SET CONFIG ###
dataset_name = args.eval_dataset
mode = "train"
input_keys = ["uv","z","rotation","grasp_state"]
input_dims = [2, 1, 6, 1]
rot_mode = "6d"
frame = 100
device = args.device

# keep configs during training
base_yamlname = os.path.basename(args.config_path)
head, ext = os.path.splitext(args.config_path)
dt_now = datetime.datetime.now()
temp_yaml_path = f"{head}_{dt_now.year}{dt_now.month}{dt_now.day}_{dt_now.hour}:{dt_now.minute}:{dt_now.second}{ext}"
shutil.copy(os.path.abspath(args.config_path), temp_yaml_path)

config_path = temp_yaml_path
cfg.merge_from_file(config_path)

for task_index, task_name in enumerate(args.tasks):

    dataset_path = f"../dataset/{dataset_name}/{mode}/{task_name}"
    print(f"dataset path:{dataset_path}")

    current_task = task_name
    print('task_name: {}'.format(current_task))

    exec_code = 'task = {}'.format(current_task)
    exec(exec_code)

    # set up task
    task = env.get_task(task)
    descriptions, obs = task.reset()

    cfg.DATASET.RLBENCH.PATH = os.path.abspath(f'../dataset/{dataset_name}')
    cfg.DATASET.RLBENCH.TASK_NAME = task_name
    eval_dataset = RLBench_DMOEBM(mode, cfg, save_dataset=False, num_frame=frame, rot_mode=rot_mode, keys=input_keys, img_aug=False)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=8)

    cfg.DATASET.RLBENCH.PATH = os.path.abspath('../dataset/RLBench4')
    train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=frame, rot_mode=rot_mode, keys=input_keys)
    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=8)
    for data in temp_loader:
        _, inf_query = data
    
    seq_list = os.listdir(dataset_path)
    seq_list.sort()

    save_dir = f"../weights/RLBench/{task_name}/{args.name}"
    result_path = os.path.join(save_dir, "result")
    os.makedirs(result_path, exist_ok=True)

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

    result_images_path = os.path.join(result_path, "image")
    result_video_path = os.path.join(result_path, "video")
    result_motion_path = os.path.join(result_path, "motion")
    result_misc_path = os.path.join(result_path, "misc")
    os.makedirs(result_images_path, exist_ok=True)
    os.makedirs(result_video_path, exist_ok=True)
    os.makedirs(result_motion_path, exist_ok=True)
    os.makedirs(result_misc_path, exist_ok=True)

    shutil.copy(config_path, result_path)

    image, query = eval_dataset[args.eval_index]
    image = torch.unsqueeze(image, 0)
    image = image.to(device)

    for key in query.keys():
        query[key] = torch.unsqueeze(query[key], 0).to(device)

    seed_path = os.path.join(dataset_path, seq_list[args.eval_index], "seed.pickle")
    with open(seed_path, 'rb') as f:
        seed = pickle.load(f)
    base_dir = os.path.join(dataset_path, seq_list[args.eval_index], 'base_data')
    gt_state_list, gt_matrix_list = get_gt_pose(base_dir)
    
    descriptions, obs = task.reset_to_seed(seed)

    if check_img(image, obs):
        img1 = torchvision.transforms.ToPILImage()(image[0,:3])
        img2 = Image.fromarray(obs.front_rgb)
        pil_img = get_concat_h(img1, img2)
        pil_img.save(os.path.join(result_misc_path, f"error_image_{str(index).zfill(5)}.png"))

    pred_action = train_dataset[args.motion_index][1]
    for key in pred_action.keys():
        pred_action[key] = torch.unsqueeze(pred_action[key], 0).to(device)

    pred_action = get_pos(pred_action, eval_dataset.info_dict["data_list"][0]["camera_intrinsic"])
    query = get_pos(query, eval_dataset.info_dict["data_list"][0]["camera_intrinsic"])

    vis_img = visualize_multi_query_pos(image, [pred_action, query], eval_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
    vis_img.save(os.path.join(result_motion_path, f"pred_motion_{str(args.eval_index).zfill(5)}.png"))

    image_list = []
    image_list.append(Image.fromarray(obs.front_rgb))
    # descriptions, obs = task.reset_to_seed(seed)

    query = {}
    for key in pred_action.keys():
        query[key] = pred_action[key].cpu()

    if rot_mode == "6d":
        query = convert_rotation_6d_to_matrix([query])[0]
    
    action_list = output2action(query, obs)

    print("simulation step")
    success = False
    max_try = 10
    total_reward = 0
    try_iter = 0
    for j,action in enumerate(action_list):
        
        if try_iter > max_try:
            error = True
            break
            
        # try control robot
        try:
            obs, reward, terminate = task.step(action)
            total_reward += reward
            image_list.append(Image.fromarray(obs.front_rgb))
            error = False
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print("error: " + str(e))
            try_iter += 1
            continue
    
    if error:
        result_dict["out of control"] += 1

    if total_reward > 0.:
        success = True
        result_dict["success"] += 1
        print(f"success!! reward:{total_reward}")
    else:
        success = False
        print(f"failure!! reward:{total_reward}")
    
    ### evaluate
    # save images
    if success:
        image_seq_dir = os.path.join(result_images_path, f"{str(args.eval_index).zfill(5)}_success")
    else:
        image_seq_dir = os.path.join(result_images_path, f"{str(args.eval_index).zfill(5)}_fail")
    os.makedirs(image_seq_dir, exist_ok=True)
    for j, image_pil in enumerate(image_list):
        image_pil.save(os.path.join(image_seq_dir, f"{str(j).zfill(5)}.png"))
    
    make_video(image_list, os.path.join(result_video_path, f"{str(args.eval_index).zfill(5)}.mp4"), (256,256), success)