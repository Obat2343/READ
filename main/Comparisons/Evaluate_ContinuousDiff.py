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
parser.add_argument('--inf_method_list', required=True, nargs="*", type=str, help='a list of inf method') # random, reconstruct, retrieve_from_SPE, retrieve_from_CLIP, retrieve_from_BYOL_wo_crop, retrieve_from_SPE_wo_modification
parser.add_argument('--result_dirname', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--num_seq', type=int, default=100)
parser.add_argument('--max_try', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--off_screen', action='store_true')
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
from pycode.retrieval import Direct_Retrieval, Image_Based_Retrieval_SPE, BYOL_Retrieval, CLIP_Retrieval, MSE_Based_Retrieval
from pycode.READ.model import SPE_Continuous_Diffusion, Timm_Continuous_Diffusion, AvgPool_Continuous_Diffusion, ConvPool_Continuous_Diffusion
from pycode.READ import sampling
from pycode.READ import sde_lib

### SET CONFIG ###
dataset_name = "RLBench-test"
mode = "val"
input_keys = ["uv","z","rotation","grasp_state"]
input_dims = [2, 1, 6, 1]
rot_mode = "6d"
max_index = args.num_seq
max_try = args.max_try
device = args.device
inference_method_list = args.inf_method_list

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
        model_config_path = os.path.join(model_path[:model_path.find("/model")], "Train_ContinuousDiff.yaml")
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
    model_name = cfg.MODEL.NAME

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

    if model_name == "Convnext-UNet":
        model = SPE_Continuous_Diffusion(input_keys, input_dims,
                    dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                    query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
    elif model_name == "Convnext-UNet-avgpool":
        model = AvgPool_Continuous_Diffusion(input_keys, input_dims,
                        dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                        query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
    elif model_name == "Convnext-UNet-convpool":
        model = ConvPool_Continuous_Diffusion(input_keys, input_dims,
                        dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                        query_emb_dim=query_emb_dim, drop_path_rate=conv_droppath_rate)
    else:
        model = Timm_Continuous_Diffusion(input_keys, input_dims, model_name, img_size=256, input_dim=4, pretrained=False,
                        query_emb_dim=query_emb_dim)
    model = model.to(device)
                    
    print(model)
    print("loading model")
    model, _, _, _, _ = load_checkpoint(model, model_path)
    model.to(device)
    model.eval()

    # set sde
    sde_name = cfg.SDE.NAME
    if sde_name == 'vpsde':
        sde = sde_lib.VPSDE(keys=input_keys, beta_min=cfg.SDE.VPSDES.BETA_MIN, beta_max=cfg.SDE.VPSDES.BETA_MAX, N=cfg.SDE.N)
        sampling_eps = 1e-3
    elif sde_name == 'subvpsde':
        sde = sde_lib.subVPSDE(keys=input_keys, beta_min=cfg.SDE.VPSDES.BETA_MIN, beta_max=cfg.SDE.VPSDES.BETA_MAX, N=cfg.SDE.N)
        sampling_eps = 1e-3
    elif sde_name == 'vesde':
        sde = sde_lib.VESDE(keys=["coordinate"], sigma_min=cfg.SDE.VESDES.SIGMA_MIN, sigma_max=cfg.SDE.VESDES.SIGMA_MAX, N=cfg.SDE.N)
        sampling_eps = 1e-5
    elif sde_name == 'irsde':
        sde = sde_lib.IRSDE(keys=input_keys, lamda=cfg.SDE.IRSDES.LAMDA, schedule="linear")
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {sde_name} unknown.")

    ################################################################################
    ### Configurations are re-changed to that of test.yaml.
    ################################################################################

    cfg.merge_from_file(config_path)
    seq_list = os.listdir(dataset_path)
    seq_list.sort()

    model_name_path = model_path[:model_path.find("/model/")]
    result_base_path = os.path.join(model_name_path, "result")
    os.makedirs(result_base_path, exist_ok=True)

    # set inference_config
    shape_dict = {}
    for key, dim in zip(input_keys, input_dims):
        # TODO change 1 to the number of samples that you want
        shape_dict[key] = (1, 101, dim)
    inverse_scaler = torch.nn.Identity()
    sampling_fn = sampling.get_sampling_fn(cfg, sde, shape_dict, inverse_scaler, sampling_eps)

    # TODO Change inference_method_list to inference_yaml_list
    for inference_method in inference_method_list:

        if args.result_dirname != "":
            result_dir_name = args.result_dirname
        else:
            if "retrieve" in inference_method:
                result_dir_name = f"{inference_method}_top{cfg.RETRIEVAL.RANK}_{sde_name}_{cfg.SDE.SAMPLING.METHOD}"
            else:
                result_dir_name = f"{inference_method}_{sde_name}_{cfg.SDE.SAMPLING.METHOD}"

            if cfg.SDE.SAMPLING.METHOD == "pc":
                result_dir_name = f"{result_dir_name}_{cfg.SDE.SAMPLING.PREDICTOR}_{cfg.SDE.SAMPLING.CORRECTOR}"
        
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
            if inference_method == "random":
                pred_action, history, n = sampling_fn(model, condition=image, history=True, N=cfg.SDE.N)
                pred_action = get_pos(pred_action, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
            elif inference_method == "reconstruct":
                pred_action, n = sampling_fn(model, x=query, condition=image, latent=False)
                pred_action = get_pos(pred_action, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
            elif "retrieve" in inference_method:
                
                reconst_steps = cfg.DIFFUSION.STEP_EVAL

                if index == 0:
                    if "retrieve_from_motion" in inference_method:
                        Retriever = Direct_Retrieval(train_dataset)
                    elif "retrieve_from_MSE" in inference_method:
                        Retriever = MSE_Based_Retrieval(train_dataset, model)
                    elif "retrieve_from_SPE" in inference_method: # changed from retrieve_from_image to retrieve_from_SPE
                        Retriever = Image_Based_Retrieval_SPE(train_dataset, model)
                    elif "retrieve_from_CLIP" in inference_method:
                        Retriever = CLIP_Retrieval(train_dataset)
                
                if inference_method == "retrieve_from_motion":
                    near_queries = Retriever.retrieve_k_sample(query)[0]
                else:
                    near_queries = Retriever.retrieve_k_sample(image)[0]

                retrieved_query = {}
                for key in near_queries.keys():
                    retrieved_query[key] = near_queries[key][:,cfg.RETRIEVAL.RANK-1].to(device)
                
                if "wo_modification" in inference_method:
                    pred_action = retrieved_query
                else:
                    pred_action, history, n = sampling_fn(model, x=retrieved_query, condition=image, history=True, N=cfg.SDE.N)
            else:
                raise ValueError("Invalid method")
            print("prediction end")

            pred_action = get_pos(pred_action, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
            query = get_pos(query, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])

            # save pred motion
            if "retrieve" in inference_method:
                near_query_list = []
                for rank_index in range(8):
                    temp = {}
                    for key in near_queries.keys():
                        temp[key] = near_queries[key][:,rank_index]
                    near_query_list.append(temp)
                near_query_list = [get_pos(temp_query, val_dataset.info_dict["data_list"][0]["camera_intrinsic"]) for temp_query in near_query_list]
                nears_img = visualize_multi_query_pos(image, near_query_list, val_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
                nears_img.save(os.path.join(result_motion_path, f"retrieved_motion_{str(index).zfill(5)}.png"))

                retrieved_query = get_pos(retrieved_query, val_dataset.info_dict["data_list"][0]["camera_intrinsic"])
                vis_img = visualize_multi_query_pos(image, [retrieved_query, pred_action, query], val_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
            else:
                vis_img = visualize_multi_query_pos(image, [pred_action, query], val_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)

            # vis_img = visualize_inf_query(vis_sample, 1, sample, query, image, train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode, pred_score=query_pred_dict["score"], gt_score=gt_pred_dict["score"])
            vis_img.save(os.path.join(result_motion_path, f"pred_motion_{str(index).zfill(5)}.png"))
        
            # if "wo_modification" not in inference_method:
            #     history_list = []
            #     history_len = len(history["x_mean"])
            #     for his_index in range(0, 10):
            #         history_list.append(get_pos(history["x_mean"][his_index], val_dataset.info_dict["data_list"][0]["camera_intrinsic"]))
            #     his_img = visualize_multi_query_pos(image, history_list, val_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
            #     his_img.save(os.path.join(result_motion_path, f"history_motion_{str(index).zfill(5)}.png"))

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