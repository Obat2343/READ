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
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--tasks', nargs="*", type=str, required=True)
parser.add_argument('--result_dirname', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--num_seq', type=int, default=100)
parser.add_argument('--max_try', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--off_screen', action='store_true')
parser.add_argument('--config_path', type=str, default="../config/Test_config.yaml")

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
from pycode.retrieval import Direct_Retrieval, Image_Based_Retrieval_SPE, BYOL_Retrieval, CLIP_Retrieval, MSE_Based_Retrieval
from pycode.Comparisons.deterministic import Deterministic_planner, Deterministic_planner_Resnet

### SET CONFIG ###
dataset_name = "RLBench-test"
mode = "val"
input_keys = ["uv","z","rotation","grasp_state"]
input_dims = [2, 1, 6, 1]
rot_mode = "6d"
max_index = args.num_seq
max_try = args.max_try
device = args.device

# keep configs during training
base_yamlname = os.path.basename(args.config_path)
head, ext = os.path.splitext(args.config_path)
dt_now = datetime.datetime.now()
temp_yaml_path = f"{head}_{dt_now.year}{dt_now.month}{dt_now.day}_{dt_now.hour}:{dt_now.minute}:{dt_now.second}{ext}"
shutil.copy(os.path.abspath(args.config_path), temp_yaml_path)

config_path = temp_yaml_path

for task_index, task_name in enumerate(args.tasks):
    if task_index == 0:
        model_path = args.model_path
        model_config_path = os.path.join(model_path[:model_path.find("/model")], "Train_Deterministic.yaml")
    else:
        pre_task_name = args.tasks[task_index - 1]
        model_path = model_path.replace(pre_task_name, task_name)
        model_config_path  = model_config_path.replace(pre_task_name, task_name)

    print(f"model path: {model_path}")
    print(f"model config path: {model_config_path}")
    print("")

    checkpoint_base_path = model_path[:model_path.find("/model")]
    argfile_path = os.path.join(checkpoint_base_path, "args.json")
    with open(argfile_path) as f:
        arg_info = json.load(f)

    frame = arg_info["frame"]

    task_dir = os.path.split(checkpoint_base_path)[0]
    print(task_name)
    # my_task = 'PickUpCup' # 'ScoopWithSpatula','ReachTarget','TakePlateOffColoredDishRack','StackWine','CloseBox','PushButton','PutKnifeOnChoppingBoard','PutRubbishInBin','PickUpCup','OpenWineBottle', 'OpenGrill', 'OpenJar', 'CloseJar', 'WipeDesk','TakePlateOffColoredDishRack', 'PutUmbrellaInUmbrellaStand'

    dataset_path = f"../dataset/{dataset_name}/{mode}/{task_name}"
    print(f"dataset path:{dataset_path}")

    current_task = task_name
    print('task_name: {}'.format(current_task))

    exec_code = 'task = {}'.format(current_task)
    exec(exec_code)

    # set up task
    task = env.get_task(task)
    descriptions, obs = task.reset()


    ################################################################################
    ### Configurations are changed to load the model and the sde.
    ################################################################################
    
    cfg.merge_from_file(model_config_path)

    dataset_name = "RLBench-test"
    cfg.DATASET.RLBENCH.PATH = os.path.abspath(f'../dataset/{dataset_name}')
    cfg.DATASET.RLBENCH.TASK_NAME = task_name
    val_dataset = RLBench_DMOEBM(mode, cfg, save_dataset=False, num_frame=frame, rot_mode=rot_mode, keys=input_keys)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    cfg.DATASET.RLBENCH.PATH = os.path.abspath('../dataset/RLBench4')
    train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=frame, rot_mode=rot_mode, keys=input_keys)
    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=8)
    for data in temp_loader:
        _, inf_query = data
        
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

    model_name = cfg.MODEL.NAME
    if model_name == "resnet18":
        model = Deterministic_planner_Resnet(input_keys, input_dims, img_size=256, predictor='transformer', model_name='resnet18')
    elif model_name == "resnet18_mlp":
        model = Deterministic_planner_Resnet(input_keys, input_dims, img_size=256, predictor='mlp', model_name='resnet18')
    else:
        model = Deterministic_planner(input_keys, input_dims,
            dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
            query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)

    print(model)
    print("loading model")
    model, _, _, _, _ = load_checkpoint(model, model_path)
    model.to(device)
    model.eval()

    ################################################################################
    ### Configurations are re-changed to that of test.yaml.
    ################################################################################

    cfg.merge_from_file(config_path)
    seq_list = os.listdir(dataset_path)
    seq_list.sort()

    model_name_path = model_path[:model_path.find("/model/")]
    result_base_path = os.path.join(model_name_path, "result")
    os.makedirs(result_base_path, exist_ok=True)

    if args.result_dirname != "":
        result_dir_name = args.result_dirname
    else:
        result_dir_name = f"Deterministic"
    
    if args.add_name != "":
        result_dir_name = f"{result_dir_name}_{args.add_name}"

    print(result_dir_name)
    result_path = os.path.join(result_base_path, result_dir_name)
    
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

    result_dict = {}
    result_dict['success'] = 0
    result_dict["out of control"] = 0
    result_dict["pose_euclid_xyz"] = []
    result_dict["pose_euclid_x"] = []
    result_dict["pose_euclid_y"] = []
    result_dict["pose_euclid_z"] = []
    result_dict["angle_euclid_xyz"] = []
    result_dict["angle_euclid_x"] = []
    result_dict["angle_euclid_y"] = []
    result_dict["angle_euclid_z"] = []
    result_dict["pose_error_list_xyz"] = []
    result_dict["pose_error_list_x"] = []
    result_dict["pose_error_list_y"] = []
    result_dict["pose_error_list_z"] = []
    result_dict["angle_error_list_xyz"] = []
    result_dict["angle_error_list_x"] = []
    result_dict["angle_error_list_y"] = []
    result_dict["angle_error_list_z"] = []
    result_dict["grasp_euclid"] = []
    result_dict["grasp_euclid_list"] = []
    
    calculation_error = False

    for index in range(max_index):
        print(f"\n{index + 1}/{max_index}")
        image, query = val_dataset[index]
        image = torch.unsqueeze(image, 0)
        image = image.to(device)

        for key in query.keys():
            query[key] = torch.unsqueeze(query[key], 0).to(device)

        seed_path = os.path.join(dataset_path, seq_list[index], "seed.pickle")
        with open(seed_path, 'rb') as f:
            seed = pickle.load(f)
        base_dir = os.path.join(dataset_path, seq_list[index], 'base_data')
        gt_state_list, gt_matrix_list = get_gt_pose(base_dir)
        
        descriptions, obs = task.reset_to_seed(seed)

        if check_img(image, obs):
            img1 = torchvision.transforms.ToPILImage()(image[0,:3])
            img2 = Image.fromarray(obs.front_rgb)
            pil_img = get_concat_h(img1, img2)
            pil_img.save(os.path.join(result_misc_path, f"error_image_{str(index).zfill(5)}.png"))

        print("prediction start")
        # get sample and score
        with torch.no_grad():
            pred_action = model(image)
            pred_action = get_pos(pred_action, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
        print("prediction end")

        query = get_pos(query, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
        vis_img = visualize_multi_query_pos(image, [pred_action, query], val_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
        vis_img.save(os.path.join(result_motion_path, f"pred_motion_{str(index).zfill(5)}.png"))

        # simulation
        if rot_mode == "6d":
            query = convert_rotation_6d_to_matrix([query])[0]

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
        """
        Note:
        pose_error_xyz = mean(pose_error_list)
        """
        try:
            pose_error_xyz, pose_error_x, pose_error_y, pose_error_z, pose_error_list_xyz, pose_error_list_x, pose_error_list_y, pose_error_list_z = calculate_euclid_pos(action_list, gt_state_list)
            angle_error_xyz, angle_error_x, angle_error_y, angle_error_z, angle_error_list_xyz, angle_error_list_x, angle_error_list_y, angle_error_list_z= calculate_euclid_angle(action_list, gt_state_list)
            grasp_error, grasp_error_list = calculate_euclid_grasp(action_list, gt_state_list)
            
            result_dict["pose_euclid_xyz"].append(pose_error_xyz)
            result_dict["pose_euclid_x"].append(pose_error_x)
            result_dict["pose_euclid_y"].append(pose_error_y)
            result_dict["pose_euclid_z"].append(pose_error_z)
            result_dict["angle_euclid_xyz"].append(angle_error_xyz)
            result_dict["angle_euclid_x"].append(angle_error_x)
            result_dict["angle_euclid_y"].append(angle_error_y)
            result_dict["angle_euclid_z"].append(angle_error_z)
            result_dict["grasp_euclid"].append(grasp_error)
            result_dict["pose_error_list_xyz"].append(pose_error_list_xyz)
            result_dict["pose_error_list_x"].append(pose_error_list_x)
            result_dict["pose_error_list_y"].append(pose_error_list_y)
            result_dict["pose_error_list_z"].append(pose_error_list_z)
            result_dict["angle_error_list_xyz"].append(angle_error_list_xyz)
            result_dict["angle_error_list_x"].append(angle_error_list_x)
            result_dict["angle_error_list_y"].append(angle_error_list_y)
            result_dict["angle_error_list_z"].append(angle_error_list_z)
            result_dict["grasp_euclid_list"].append(grasp_error_list)
        except ValueError:
            calculation_error = True
        
        # save images
        if success:
            image_seq_dir = os.path.join(result_images_path, f"{str(index).zfill(5)}_success")
        else:
            image_seq_dir = os.path.join(result_images_path, f"{str(index).zfill(5)}_fail")
        os.makedirs(image_seq_dir, exist_ok=True)
        for j, image_pil in enumerate(image_list):
            image_pil.save(os.path.join(image_seq_dir, f"{str(j).zfill(5)}.png"))
        
        make_video(image_list, os.path.join(result_video_path, f"{str(index).zfill(5)}.mp4"), (256,256), success)

    print("save_result to csv")    
    # save_result
    csv_path = os.path.join(result_path, "result.csv")
    head_list = ["main_model_path", "sub_model_path", "date", "mode", "num_try", "num_succes",
                "pose_euclid_xyz","pose_euclid_x","pose_euclid_y","pose_euclid_z",
                "angle_euclid_xyz","angle_euclid_x","angle_euclid_y","angle_euclid_z",
                "num of out of control", "\n"]

    dt_now = datetime.datetime.now()
    if calculation_error:
        list_to_csv = [model_path, "none", dt_now, mode, max_index, result_dict["success"],
            "none", "none",
            "none", "none",
            "none", "none",
            "none", "none",
            result_dict["out of control"]]
    else:
        list_to_csv = [model_path, "none", dt_now, mode, max_index, result_dict["success"],
            np.mean(result_dict["pose_euclid_xyz"]), np.mean(result_dict["pose_euclid_x"]),
            np.mean(result_dict["pose_euclid_y"]), np.mean(result_dict["pose_euclid_z"]),
            np.mean(result_dict["angle_euclid_xyz"]), np.mean(result_dict["angle_euclid_x"]),
            np.mean(result_dict["angle_euclid_y"]), np.mean(result_dict["angle_euclid_z"]),
            result_dict["out of control"]]

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(head_list)
        writer.writerow(list_to_csv)