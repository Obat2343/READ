import os
import csv
import sys
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
from einops import rearrange, reduce, repeat


##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--csv_path', type=str, default="")
parser.add_argument('--camera_path', type=str, default="")
parser.add_argument('--save_path', type=str, default="")
args = parser.parse_args()

# load pose
times, left_poses, left_rotations, left_gripper_states, right_poses, right_rotations, right_gripper_states = [], [], [], [], [], [], []
with open(args.csv_path, 'r') as csv_path:
    reader = csv.reader(csv_path)

    # Read and modify header
    header = next(reader)

    for row in reader:
        timestamp = float(row[0])

        left_position = list(map(float, row[1:4]))  # posx, posy, posz
        left_orientation = list(map(float, row[4:8]))  # qx, qy, qz, qw
        left_grip = float(row[8])  # Gripper position

        right_position = list(map(float, row[9:12]))  # posx, posy, posz
        right_orientation = list(map(float, row[12:16]))  # qx, qy, qz, qw
        right_grip = float(row[16])  # Gripper position

        times.append(timestamp)
        left_poses.append(left_position)
        left_rotations.append(left_orientation)
        left_gripper_states.append(left_grip)
        right_poses.append(right_position)
        right_rotations.append(right_orientation)
        right_gripper_states.append(right_grip)

times = np.array(times)
left_poses = np.array(left_poses)
# left_rotations = R.from_quat(np.array(left_rotations))
left_rotations = np.array(left_rotations)
left_gripper_states = np.array(left_gripper_states)
right_poses = np.array(right_poses)
# right_rotations = R.from_quat(np.array(right_rotations))
right_rotations = np.array(right_rotations)
right_gripper_states = np.array(right_gripper_states)

# get camera info
camera_intrinsic = np.load(args.camera_path)

# image size
W, H = 1280, 800
image_size = (H, W)

def get_uv(pos_data, intrinsic_matrix):
    # transfer position data(based on motive coordinate) to camera coordinate
    B, _ = pos_data.shape
    z = np.repeat(pos_data[:, 2], 3).reshape((B,3))
    pos_data = pos_data / z # u,v,1
    uv_result = np.einsum('ij,bj->bi', intrinsic_matrix, pos_data)
    return uv_result[:, :2]

# get uv cordinate and pose image
left_uv = get_uv(left_poses, camera_intrinsic)
right_uv = get_uv(right_poses, camera_intrinsic)

action = np.concatenate([times[:,None], left_uv, left_poses[:, 2:3], left_rotations, left_gripper_states[:,None], right_uv, right_poses[:, 2:3], right_rotations, right_gripper_states[:,None]], 1)

# save_result
print("convert")    
# head_list = ["Time", "left_u", "left_v", "left_z", "left_qx", "left_qy", "left_qz", "left_qw", "left_grip", "right_u", "right_v", "right_z", "right_qx", "right_qy", "right_qz", "right_qw", "right_grip"]
header = "Time,left_u,left_v,left_z,left_qx,left_qy,left_qz,left_qw,left_grip,right_u,right_v,right_z,right_qx,right_qy,right_qz,right_qw,right_grip"

np.savetxt(args.save_path, action, delimiter=",", fmt="%.11f", header=header, comments="")