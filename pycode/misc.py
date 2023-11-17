import io
import os
import time
import json
import pickle

import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

import cv2
import torch
import torchvision
import pytorch3d
import pytorch3d.transforms
import torch.nn as nn

from typing import Dict, Callable, List
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from scipy import interpolate
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange, reduce, repeat
from torchvision import models

from fastdtw import fastdtw # https://github.com/slaypni/fastdtw
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

##### timer #####

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs

class Time_memo(object):
    def __init__(self):
        self.value_dict = {}
        self.count_dict = {}

    def add(self, key, value):
        if key not in self.value_dict.keys():
            self.value_dict[key] = value
            self.count_dict[key] = 1
        else:
            self.value_dict[key] += value
            self.count_dict[key] += 1

    def get(self, key):
        if key not in self.value_dict.keys():
            print("can not find key.")
            return 0
        else:
            return self.value_dict[key] / self.count_dict[key]

    def reset(self):
        self.value_dict = {}
        self.count_dict = {}


##### save model #####

def save_args(args,file_path="args_data.json"):
    with open(file_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def save_checkpoint(model, optimizer, epoch, iteration, file_path, scheduler=False):
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch'] = epoch
    checkpoint['iteration'] = iteration
    if scheduler != False:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint,file_path)

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        if 'norm' in name:
            start_index = name.find('norm')
            name = name[:start_index] + 'pose' + name[start_index+4:]
        new_state_dict[name] = v
    return new_state_dict

def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None,fix_parallel=False):
    checkpoint = torch.load(checkpoint_path)
    if fix_parallel:
        print('fix parallel')
        model.load_state_dict(fix_model_state_dict(checkpoint['model']), strict=True)
    else:
        model.load_state_dict(checkpoint['model'], strict=True)
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return model, optimizer, epoch, iteration, scheduler

##### Visualization #####

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width + 1, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + 1, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + 1))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + 1))
    return dst

def visualize(x, debug_info, gt, save_dir, iteration):
    visualize_dict = {}
    os.makedirs(save_dir, exist_ok=True)
    for key in debug_info.keys():
        if "heatmap" in key:
            visualize_dict[key] = visualize_heatmap(x, gt, debug_info[key], debug_info["pose"])
        elif key == "uv":
            visualize_dict[key] = visualize_points(x, gt, debug_info[key], debug_info["pose"])
        elif "atten_points" in key:
            visualize_dict[key] = visualize_points(x, gt, debug_info[key], debug_info["pose"])
        elif "atten_mask" in key:
            visualize_dict[key] = visualize_heatmap(x, gt, debug_info[key], debug_info["pose"])
        elif "energy_map" in key:
            visualize_dict[key] = visualize_valuemap(x, key, debug_info, save_dir, iteration)
        elif "pred_coef" == key:
            visualize_dict[key] = visualize_points(x, gt, debug_info["pred_uv"], coef = debug_info[key])
        elif key[:3] == "sep":
            visualize_dict[key] = visualize_valuemap(x, key, debug_info, save_dir, iteration)
        elif "pose" in key:
            continue
        else:
            print(f"visualize {key} is not implemented")
            
    return visualize_dict

def visualize_points(x, gt, pred, points, r=6, coef='none', img_size=(128,128)):
    x, gt, pred = x.cpu(), gt.cpu(), pred.detach().cpu()
    B, _, H, W = x.shape
    _, P, _ = points.shape
    r = 3
    
    for B_index in range(B):
        tensor_img = x[B_index]
        pil_image = torchvision.transforms.ToPILImage()(tensor_img)
        gt_image = pil_image.copy()
        draw = ImageDraw.Draw(gt_image)
        u, v = gt[B_index].tolist()
        u, v = round(u), round(v)
        draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(0, 0, 255))
        
        u, v = pred[B_index].tolist()
        u, v = round(u), round(v)
        draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(255, 0, 0))
        
        img = get_concat_h(pil_image.resize(img_size), gt_image.resize(img_size))
        
        for n_index in range(P):
            pil_image_for_draw = pil_image.copy()
            draw = ImageDraw.Draw(pil_image_for_draw)
            u, v = points[B_index, n_index].tolist()
            u, v = round(u), round(v)
            
            draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(0, 0, 0))
            
            pil_image_for_draw = pil_image_for_draw.resize(img_size)
            draw = ImageDraw.Draw(pil_image_for_draw)
            if coef != 'none':
                _, Num_query, Num_heads, _ = coef.shape
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 12, encoding="unic")
                coef_value = torch.sum(coef[B_index,:,:,n_index]).item() / (Num_heads * Num_query) # B N h k
                draw.text((5,5), "coef: {:.2f}".format(coef_value), (0, 0, 0), font)
                
            if n_index == 0:
                point_image_h = get_concat_h(img, pil_image_for_draw)
            else:
                point_image_h = get_concat_h(point_image_h, pil_image_for_draw)

        if B_index == 0:
            point_image_v = point_image_h
        else:
            point_image_v = get_concat_v(point_image_v, point_image_h)
    
    return point_image_v

def visualize_heatmap(x, gt, pred, heatmaps, r=6, img_size=(128,128)):
    x, gt, pred = x.cpu(), gt.cpu(), pred.detach().cpu()
    B, _, H, W = x.shape
    _, C_heat, _, _ = heatmaps.shape
    heatmaps = torch.nn.functional.interpolate(heatmaps, size=(H,W), mode='bicubic', align_corners=True)
    heatmaps = repeat(heatmaps, 'b n h w -> b n c h w', c=3)
    
    temp = rearrange(heatmaps, 'b n c h w -> (b n) (c h w)')
    max_values, index = torch.max(temp, 1)
    max_values = rearrange(max_values, '(b n) -> b n', b=B)
    heatmaps = heatmaps / repeat(max_values, 'b n -> b n c h w', c=3, h=H, w=W)

    for B_index in range(B):
        tensor_img = x[B_index]
        pil_image = torchvision.transforms.ToPILImage()(tensor_img)
        gt_image = pil_image.copy()
        draw = ImageDraw.Draw(gt_image)
        u, v = gt[B_index].tolist()
        u, v = round(u), round(v)
        draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(0, 0, 255))
        
        u, v = pred[B_index].tolist()
        u, v = round(u), round(v)
        draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(255, 0, 0))
        
        img = get_concat_h(pil_image.resize(img_size), gt_image.resize(img_size))
        
        for n_index in range(C_heat):
            overlay_tensor = (tensor_img * 0.3) + (heatmaps[B_index, n_index] * 0.7)
            overlay_img = torchvision.transforms.ToPILImage()(overlay_tensor)

            if n_index == 0:
                concat_image_h = get_concat_h(img, overlay_img.resize(img_size))
            else:
                concat_image_h = get_concat_h(concat_image_h, overlay_img.resize(img_size))

        if B_index == 0:
            concat_v = concat_image_h
        else:
            concat_v = get_concat_v(concat_v, concat_image_h)
    
    return concat_v

def visualize_valuemap(x, key, debug_info, save_dir, iteration, r=6, img_size=(128,128)):
    heatmap_dict = debug_info[key]
    pred_pose_dict = debug_info["pred_pose"]
    gt_pose_dict = debug_info["gt_pose"]
    x = x.cpu()
    B, _, H, W = x.shape

    for time_index, heatmaps in enumerate(heatmap_dict["value"]):
        pred_pose = pred_pose_dict["value"][time_index]
        time = heatmap_dict["time"][time_index]
        gt = gt_pose_dict["value"][time_index]
        
        _, C_heat, _, _ = heatmaps.shape
        heatmaps = repeat(heatmaps, 'b n h w -> b (n c) h w', c=3)

        temp = rearrange(heatmaps, 'b c h w -> b (c h w)')
        min_values, index = torch.min(temp, 1)
        heatmaps_n = heatmaps - repeat(min_values, 'b -> b c h w', c=3, h=H, w=W)
        
        temp = rearrange(heatmaps_n, 'b c h w -> b (c h w)')
        max_values, index = torch.max(temp, 1)
        heatmaps_n = heatmaps_n / repeat(max_values, 'b -> b c h w', c=3, h=H, w=W)
        
        heatmaps_n = 1 - heatmaps_n

        xx_ones = torch.arange(W, dtype=torch.int32)
        xx_channel = repeat(xx_ones, 'W -> H W', H=H).numpy()

        yy_ones = torch.arange(H, dtype=torch.int32)
        yy_channel = repeat(-yy_ones, 'H -> H W', W=W).numpy()

        for B_index in range(B):
            tensor_img = x[B_index]
            pil_image = torchvision.transforms.ToPILImage()(tensor_img)
            
            gt_image = pil_image.copy()
            draw = ImageDraw.Draw(gt_image)
            u, v = gt[B_index].tolist()
            u, v = round(u), round(v)
            draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(0, 0, 255))
            
            u, v = pred_pose[B_index].tolist()
            u, v = round(u), round(v)
            draw.arc((u - r, v - r, u + r, v + r), start=0, end=360, fill=(255, 0, 0))
            
            img = get_concat_h(pil_image.resize(img_size), gt_image.resize(img_size))

            overlay_tensor = (tensor_img * 0.3) + (heatmaps_n[B_index] * 0.7)
            overlay_img = torchvision.transforms.ToPILImage()(overlay_tensor)

            concat_image_h = get_concat_h(img, overlay_img.resize(img_size))
            
            # get contour_img
            fig = plt.figure(figsize=(4, 4), dpi=100)
            cont = plt.contour(xx_channel,yy_channel, heatmaps[B_index,0].numpy(), 8, colors=['r', 'g', 'b'])
            cont.clabel(fmt='%1.1f', fontsize=20)
            plt.axis('off')
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png',bbox_inches='tight',pad_inches = 0)
            
            contour_img = Image.open(img_buf).resize(img_size)
            plt.clf()
            plt.close()
            
            temp_img = pil_image.copy().resize(img_size)
            temp_img.paste(contour_img, (0, 0), contour_img)
            concat_image_h = get_concat_h(concat_image_h, temp_img)

            if B_index == 0:
                concat_v = concat_image_h
            else:
                concat_v = get_concat_v(concat_v, concat_image_h)
        
        concat_v.save(os.path.join(save_dir, "{}_iter{}_time{}.png".format(key, iteration, time)))
    
def draw_from_rot_and_pos(image, rotation_matrix, pos_vector, intrinsic_matrix):
    """
    image: PIL.Image
    pose_matrix: np.array (4X4)
        pose is position and orientation in the camera coordinate.
    intrinsic_matrix: np.array(4X4)
    """
    pose_matrix = np.append(rotation_matrix, pos_vector.T, 1)
    pose_matrix = np.append(pose_matrix, np.array([[0,0,0,1]]),0)
    
    cordinate_vector_array = np.array([[0,0,0,1],[0,0,0.1,1],[0,0.1,0,1],[0.1,0,0,1]]).T
    cordinate_matrix = np.dot(pose_matrix, cordinate_vector_array)
    
    draw = ImageDraw.Draw(image)
    color_list = [(255,0,0), (0,255,0), (0,0,255)]
    
    base_cordinate = cordinate_matrix.T[0]
    cordinates = cordinate_matrix.T[1:]

    base_cordinate = base_cordinate[:3] / base_cordinate[2]
    base_uv = np.dot(intrinsic_matrix, base_cordinate)
    base_u, base_v = base_uv[0], base_uv[1]
    
    for i in range(len(cordinates)):
        cordinate = cordinates[i]
        cordinate = cordinate[:3] / cordinate[2]
        uv = np.dot(intrinsic_matrix, cordinate)
        u, v = uv[0], uv[1]
        
        draw.line((base_u, base_v, u, v), fill=color_list[i], width=3)
    
    return image

def make_rotation_image(rgb, rotation_matrix, pose_vec, intrinsic_matrix):
    """
    input
    rgb: tensor (B, C, H, W)
    rotation_matrix: tensor
    intrinsic_matrix: tensor
    
    output
    image_sequence: tensor (BS, C, H ,W)
    """
    B, N, _ = pose_vec.shape
    _, C, H, W = rgb.shape
    topil = torchvision.transforms.ToPILImage()
    totensor = torchvision.transforms.ToTensor()
    
    for b in range(B):
        for n in range(N):
            image_pil = topil(torch.clamp(rgb[b], 0, 1))
            rotation_np = rotation_matrix[b, n].cpu().numpy()

            pos_vec_np = pose_vec[b, n].cpu().numpy()
            pos_vec_np = np.expand_dims(pos_vec_np, 0)

            intrinsic_matrix_np = intrinsic_matrix[b, n].cpu().numpy()
            image_pil = draw_matrix(image_pil, rotation_np, pos_vec_np, intrinsic_matrix_np)
            
            if n == 0:
                image_h = image_pil
            else:
                image_h = get_concat_h(image_h, image_pil)
                
        if b == 0:
            image_v = image_h
        else:
            image_v = get_concat_v(image_v, image_h)
    
    return image_v

def visualize_query(image, query, camera_intrinsic, rot_mode="quat", img_size=(128,128), score="none", skip_len=1, draw="uv"):
    """
    image: torch.tensor shape(4,H,W)
    query: dict
        uv: torch.tensor shape(N, 2)
        z: torch.tenosr shape(N, 1)
        rot: torch.tensor shape(N, 4)
    """
    pil_image = torchvision.transforms.ToPILImage()(image[:3]).copy()
    pos = query["pos"]
    rot = query["rotation"]
    
    if (rot_mode == "6d") and (rot.dim() == 2):
        rot = pytorch3d.transforms.rotation_6d_to_matrix(rot)

    pos_np = pos.numpy()
    rot_np = rot.numpy()
    
    if rot_mode in ["matrix", "6d"]:
        rot_R = R.from_matrix(rot_np)
    elif rot_mode == "quat":
        rot_R = R.from_quat(rot_np)
    else:
        raise ValueError("invalid mode")
        
    rot_matrix = rot_R.as_matrix()
    
    for i in range(0, len(pos_np), skip_len):
        pose_matrix = np.append(rot_matrix[i], pos_np[i:i+1].T, 1)
        pose_matrix = np.append(pose_matrix, np.array([[0,0,0,1]]),0)
        
        ratio = (i + 1) / len(pos_np)
        if draw == "matrix":
            pil_image = draw_matrix(pil_image, pose_matrix, camera_intrinsic, color_rate=ratio)
        else:
            pil_image = draw_center(pil_image, pose_matrix, camera_intrinsic, color_rate=ratio)
    
    pil_image.resize(img_size)
        
    if score != "none":
        textcolor = (255, 255, 255)
        # textsize = int(15 * img_size[0] / 128)

        text = f"score: {score:.4g}"
        txpos = (5, 5)

        draw_image = ImageDraw.Draw(pil_image)

        # draw multiline text
        draw_image.text(txpos, text, fill=textcolor)

    return pil_image
    
def visualize_uv(image, query, do_uv_denorm=True, r=3, img_size=(128,128)):
    """
    image: torch.tensor shape(4,H,W)
    query: dict
        uv: torch.tensor shape(N, 2)
        z: torch.tenosr shape(N, 1)
        rot: torch.tensor shape(N, 4)
    """
    C,H,W = image.shape
    pil_image = torchvision.transforms.ToPILImage()(image[:3])
    draw = ImageDraw.Draw(pil_image)
    uv = query["uv"]
    
    # denorm uv
    if do_uv_denorm:
        uv = denorm_uv(uv,(H,W))
        
    uv = uv.numpy()
    
    for i in range(len(uv)):
        u, v = uv[i, 0], uv[i, 1]
        ratio = (i + 1) / len(uv)
        draw.ellipse((u-r, v-r, u+r, v+r), fill=(int(255 * ratio), 0, 0), outline=(0, 0, 0))
    return pil_image.resize(img_size)
    
    
def draw_matrix(image, pose_matrix, intrinsic_matrix, color_rate=1.0):
    """
    image: PIL.Image
    pose_matrix: np.array (4X4)
        pose is position and orientation in the camera coordinate.
    intrinsic_matrix: np.array(4X4)
    """
    cordinate_vector_array = np.array([[0,0,0,1],[0,0,0.1,1],[0,0.1,0,1],[0.1,0,0,1]]).T
    cordinate_matrix = np.dot(pose_matrix, cordinate_vector_array)
    
    draw = ImageDraw.Draw(image)
    color = int(255 * color_rate)
    color_list = [(color,0,0), (0,color,0), (0,0,color)]
    
    base_cordinate = cordinate_matrix.T[0]
    cordinates = cordinate_matrix.T[1:]

    base_cordinate = base_cordinate[:3] / base_cordinate[2]
    base_uv = np.dot(intrinsic_matrix, base_cordinate)
    base_u, base_v = base_uv[0], base_uv[1]
    
    for i in range(len(cordinates)):
        cordinate = cordinates[i]
        cordinate = cordinate[:3] / cordinate[2]
        uv = np.dot(intrinsic_matrix, cordinate)
        u, v = uv[0], uv[1]
        
        draw.line((base_u, base_v, u, v), fill=color_list[i], width=3)
    
    return image

def draw_center(image, pose_matrix, intrinsic_matrix, color_rate=1.0, r=3):
    """
    image: PIL.Image
    pose_matrix: np.array (4X4)
        pose is position and orientation in the camera coordinate.
    intrinsic_matrix: np.array(4X4)
    """
    cordinate_vector_array = np.array([[0,0,0,1],[0,0,0.1,1],[0,0.1,0,1],[0.1,0,0,1]]).T
    cordinate_matrix = np.dot(pose_matrix, cordinate_vector_array)
    
    draw = ImageDraw.Draw(image)
    
    import colorsys

    def hsv_to_rgb(h, s=0.95, v=1.0):
        # colorsysモジュールを使用してHSVからRGBに変換
        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        # RGB値を0から255の整数に変換
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        return (r, g, b)
    
    color = hsv_to_rgb(color_rate)
    # color_list = [(color,0,0), (0,color,0), (0,0,color)]
    
    base_cordinate = cordinate_matrix.T[0]
    cordinates = cordinate_matrix.T[1:]

    base_cordinate = base_cordinate[:3] / base_cordinate[2]
    base_uv = np.dot(intrinsic_matrix, base_cordinate)
    base_u, base_v = base_uv[0], base_uv[1]
    
    draw.ellipse([base_u - r, base_v - r, base_u + r, base_v + r], fill=color)
    return image

def visualize_query_batch(img_batch, query_batch, camera_intrinsic, rot_mode="quat", img_size=(128,128), score="none", skip_len=1, draw="uv"):
    
    for i, img in enumerate(img_batch):
        mini_query = {}
        for key in query_batch.keys():
            mini_query[key] = query_batch[key][i].cpu()
        
        if score != "none":
            score_ins = score[i].item()
        else:
            score_ins = "none"

        if i == 0:
            pil_img = visualize_query(img, mini_query, camera_intrinsic, rot_mode=rot_mode, img_size=img_size, score=score_ins, skip_len=skip_len, draw=draw)
        else:
            pil_img = get_concat_h(pil_img, visualize_query(img, mini_query, camera_intrinsic, rot_mode=rot_mode, img_size=img_size, score=score_ins, skip_len=skip_len, draw=draw))
    
    return pil_img

def visualize_uv_batch(img_batch, query_batch, do_uv_denorm=True, r=3, img_size=(128,128)):
    
    for i, img in enumerate(img_batch):
        mini_query = {}
        for key in query_batch.keys():
            mini_query[key] = query_batch[key][i]
        
        if i == 0:
            pil_img = visualize_uv(img, mini_query, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size)
        else:
            pil_img = get_concat_h(pil_img, visualize_uv(img, mini_query, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size))
    
    return pil_img

def visualize_two_query_all(img_batch, query_batch1, query_batch2, camera_intrinsic, rot_mode="quat", do_uv_denorm=True, r=3, img_size=(128,128)):
    # get pose image1
    pil_img = visualize_query_batch(img_batch, query_batch1, camera_intrinsic, rot_mode=rot_mode, img_size=img_size)
    
    # get pose_image2
    pil_img = get_concat_v(pil_img, visualize_query_batch(img_batch, query_batch2, camera_intrinsic, rot_mode=rot_mode, img_size=img_size))
    
    # get uv image1
    pil_img = get_concat_v(pil_img, visualize_uv_batch(img_batch, query_batch1, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size))
    
    # get uv image2
    pil_img = get_concat_v(pil_img, visualize_uv_batch(img_batch, query_batch2, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size))
    
    return pil_img

def visualize_multi_query_all(img_batch, query_batch_list, camera_intrinsic, rot_mode="quat", do_uv_denorm=True, r=3, img_size=(128, 128)):
    for i, query_batch in enumerate(query_batch_list):
        if i == 0:
            pil_img = visualize_query_batch(img_batch, query_batch, camera_intrinsic, rot_mode=rot_mode, img_size=img_size)
        else:
            pil_img = get_concat_v(pil_img, visualize_query_batch(img_batch, query_batch, camera_intrinsic, rot_mode=rot_mode, img_size=img_size))
    
    for i, query_batch in enumerate(query_batch_list):
        pil_img = get_concat_v(pil_img, visualize_uv_batch(img_batch, query_batch, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size))
    
    return pil_img

def visualize_multi_query_pos(img_batch, query_batch_list, camera_intrinsic, rot_mode="quat", img_size=(128, 128), max_img_num=32, skip_len=1, score_list=[], draw="uv"):
    for i, query_batch in enumerate(query_batch_list):
        for key in query_batch.keys():
            B = query_batch[key].shape[0]

            if B > max_img_num:
                query_batch_list[i][key] = query_batch[key][:max_img_num]

    if B > max_img_num:
        img_batch = img_batch[:max_img_num]

    for i, query_batch in enumerate(query_batch_list):
        if len(score_list) != 0:
            score = score_list[i]
        else:
            score = "none"

        if i == 0:
            pil_img = visualize_query_batch(img_batch, query_batch, camera_intrinsic, rot_mode=rot_mode, img_size=img_size, score=score, skip_len=skip_len, draw=draw)
        else:
            pil_img = get_concat_v(pil_img, visualize_query_batch(img_batch, query_batch, camera_intrinsic, rot_mode=rot_mode, img_size=img_size, score=score, skip_len=skip_len, draw=draw))
    
    return pil_img

def visualize_multi_query_uv(img_batch, query_batch_list, do_uv_denorm=True, r=3, img_size=(128, 128), max_img_num=32):
    for i, query_batch in enumerate(query_batch_list):
        for key in query_batch.keys():
            B = query_batch[key].shape[0]

            if B > max_img_num:
                query_batch_list[i][key] = query_batch[key][:max_img_num]

    if B > max_img_num:
        img_batch = img_batch[:max_img_num]

    for i, query_batch in enumerate(query_batch_list):
        if i == 0:
            pil_img = visualize_uv_batch(img_batch, query_batch, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size)
        else:
            pil_img = get_concat_v(pil_img, visualize_uv_batch(img_batch, query_batch, do_uv_denorm=do_uv_denorm, r=r, img_size=img_size))
    
    return pil_img

def visualize_negative_sample(image, query, camera_intrinsic, rot_mode="6d"):
    
    def get_one_from_batch(query, index=0):
        temp_dict = {}
        for key in query.keys():
            temp_dict[key] = query[key][index].cpu()
        return temp_dict

    B, N, S, _ = query["pos"].shape
    for i in range(B):
        ins_image = repeat(image[i], "C H W -> N C H W", N=N)
        ins_query = get_one_from_batch(query, index=i)
        ins_query = convert_rotation_6d_to_matrix([ins_query])[0]
        if i == 0:
            pil_img = visualize_query_batch(ins_image, ins_query, camera_intrinsic, rot_mode=rot_mode)
        else:
            pil_img = get_concat_v(pil_img, visualize_query_batch(ins_image, ins_query, camera_intrinsic, rot_mode=rot_mode))
            
    return pil_img

def visualize_inf_query(top_n, batch_size, inf_sample, gt_query, image, intrinsic, rot_mode, pred_score="none", gt_score="none"):
    query_list = []
    score_list = []

    if gt_score != "none":
        score_list.append(gt_score[:batch_size, 0])
    else:
        score_list.append("none")

    for n in range(top_n):
        temp_query = {}
        for key in inf_sample.keys():
            temp_query[key] = inf_sample[key][:batch_size, n].cpu()
        query_list.append(temp_query)
        
        if pred_score != "none":
            score_list.append(pred_score[:batch_size, n])
        else:
            score_list.append("none")

    image = image[:batch_size].cpu()

    for key in gt_query.keys():
        gt_query[key] = gt_query[key].cpu()

    query_list.insert(0, gt_query)
    
    return visualize_multi_query_pos(image, query_list, intrinsic, rot_mode=rot_mode, score_list=score_list)

##### else #####
def cat_pos_and_neg(positive_query, negative_query, device='cuda'):
    cat_query = {}
    for key in positive_query.keys():
        cat_query[key] = torch.cat([torch.unsqueeze(positive_query[key], 1).to(device), negative_query[key].to(device)], 1)
    
    return cat_query

def gaussian_noise(query,pos_std,rot_std,grasp_prob,intrinsic,rot_mode="quat",image_size=(256,256),do_norm=True):
    # pos
    pos = query["pos"]
    shape = pos.shape
    B = shape[0]
    dim = pos.dim()
    if dim == 3:
        N = shape[1]
    elif dim == 2:
        N = 1
    else:
        raise ValueError("TODO")
    
    pos_noise = torch.normal(0., pos_std, size=shape)
    noise_pos = pos + pos_noise

    # rotation
    rot = query["rotation"]
    rot_shape = list(rot.shape)
    rot_noise = torch.normal(0., rot_std, size=(B * N, 3))
    
    if dim == 3:
        rot = rearrange(rot, "B N ... -> (B N) ...")
    
    if rot_mode == "quat":
        rot_r = R.from_quat(rot.numpy())
        rot_euler = rot_r.as_euler('zxy', degrees=True)
        noise_rot_euler = rot_euler + rot_noise.numpy()
        noise_rot_r = R.from_euler('zxy', noise_rot_euler, degrees=True)
        noise_rot = noise_rot_r.as_quat()
        noise_rot = torch.tensor(noise_rot, dtype=torch.float)
    elif rot_mode == "euler":
        noise_rot_euler = rot.numpy() + rot_noise.numpy()
        noise_rot_r = R.from_euler('zxy', noise_rot_euler, degrees=True)
        noise_rot = noise_rot_r.as_euler('zxy', degrees=True)
        noise_rot = torch.tensor(noise_rot, dtype=torch.float)
    elif rot_mode == "matrix":
        rot_r = R.from_matrix(rot.numpy())
        rot_euler = rot_r.as_euler('zxy', degrees=True)
        noise_rot_euler = rot_euler + rot_noise.numpy()
        noise_rot_r = R.from_euler('zxy', noise_rot_euler, degrees=True)
        noise_rot = noise_rot_r.as_matrix()
        noise_rot = torch.tensor(noise_rot, dtype=torch.float)
    elif rot_mode == "6d":
        rot = pytorch3d.transforms.rotation_6d_to_matrix(rot)
        rot_r = R.from_matrix(rot.numpy())
        rot_euler = rot_r.as_euler('zxy', degrees=True)
        noise_rot_euler = rot_euler + rot_noise.numpy()
        noise_rot_r = R.from_euler('zxy', noise_rot_euler, degrees=True)
        noise_rot = noise_rot_r.as_matrix()
        noise_rot = torch.tensor(noise_rot, dtype=torch.float)
        noise_rot = pytorch3d.transforms.matrix_to_rotation_6d(noise_rot)
    
    if dim == 3:
        noise_rot = rearrange(noise_rot, "(B N) ... -> B N ...", B=B)

    # grasp
    grasp = query["grasp_state"]
    weights = torch.tensor([1-grasp_prob, grasp_prob], dtype=torch.float) # create a tensor of weights
    temp = torch.unsqueeze(torch.multinomial(weights, B*N, replacement=True), 1) * -1
    if dim == 3:
        temp = rearrange(temp, "(B N) ... -> B N ...", B=B)
    noise_grasp = torch.abs(grasp + temp)

    # uv and z
    if dim == 2:
        noise_z = noise_pos[:, 2]
        z_repet = repeat(noise_z, "B -> B Z", Z=3)
        pos_data = noise_pos / z_repet # u,v,1
        noise_uv = torch.einsum('ij,bj->bi', torch.tensor(intrinsic, dtype=torch.float), pos_data)
        u, v = noise_uv[:, 0], noise_uv[:, 1]
        noise_z = torch.unsqueeze(noise_z, 1)
    elif dim == 3:
        noise_z = noise_pos[:,:,2]
        z_repet = repeat(noise_z, "B N -> B N Z", Z=3)
        pos_data = noise_pos / z_repet # u,v,1
        noise_uv = torch.einsum('ij,bkj->bki', torch.tensor(intrinsic, dtype=torch.float), pos_data)
        u, v = noise_uv[:, :, 0], noise_uv[:, :, 1]
        noise_z = torch.unsqueeze(noise_z, 2)

    if do_norm == True:
        h, w = image_size
        u = (u / (w - 1) * 2) - 1
        v = (v / (h - 1) * 2) - 1
        
        if dim == 2:
            noise_uv = torch.stack([u, v], 1)
        elif dim == 3:
            noise_uv = torch.stack([u, v], 2)
        
    noise_query = {}
    noise_query["pos"] = noise_pos
    noise_query["rotation"] = noise_rot
    noise_query["grasp_state"] = noise_grasp
    noise_query["uv"] = noise_uv
    noise_query["z"] = noise_z
    noise_query["time"] = query["time"]
    return noise_query

def interpolate_batch(query, output_time, rot_mode="quat"):
    interpolated_query = {}
    time_batch = query["time"][0,:,0].tolist()
    
    # pos
    pos_batch = query["pos"].cpu().numpy()
    pos_batch = pos_batch.transpose((2,0,1))
    pos_curve = interpolate.interp1d(time_batch, pos_batch, kind="cubic", fill_value="extrapolate")
    interpolated_pos = pos_curve(output_time).transpose((1,2,0))
    
    # rot
    rotation_batch = query["rotation"]
    B, N, D = rotation_batch.shape

    interpolated_rot = []
    for i in range(B):
        if rot_mode == "quat":
            query_rot = rotation_batch[i].cpu().numpy()
            query_rot = R.from_quat(query_rot)
            rot_curve = RotationSpline(time_batch, query_rot)
            interpolated_rot_ins = rot_curve(output_time).as_quat()
        elif rot_mode == "euler":
            query_rot = rotation_batch[i].cpu().numpy()
            query_rot = R.from_euler('zxy', query_rot, degrees=True)
            rot_curve = RotationSpline(time_batch, query_rot)
            interpolated_rot_ins = rot_curve(output_time).as_quat('zxy', degrees=True)
        elif rot_mode == "matrix":
            query_rot = R.from_matrix(query_rot.cpu().numpy())
            rot_curve = RotationSpline(time_batch, query_rot)
            interpolated_rot_ins = rot_curve(output_time).as_matrix()
        elif rot_mode == "6d":
            query_rot = rotation_batch[i]
            query_rot = pytorch3d.transforms.rotation_6d_to_matrix(query_rot)
            query_rot = R.from_matrix(query_rot.cpu().numpy())
            rot_curve = RotationSpline(time_batch, query_rot)
            interpolated_rot_ins = rot_curve(output_time).as_matrix()
        interpolated_rot.append(interpolated_rot_ins)
    
    # grasp
    grasp_state_batch = query["grasp_state"].cpu().numpy()
    grasp_state_batch = grasp_state_batch.transpose((2,0,1))
    grasp_curve = interpolate.interp1d(time_batch, grasp_state_batch, fill_value="extrapolate")
    interpolated_grasp = grasp_curve(output_time).transpose((1,2,0))
    
    # uv
    uv_batch = query["uv"].cpu().numpy()
    uv_batch = uv_batch.transpose((2,0,1))
    uv_curve = interpolate.interp1d(time_batch, uv_batch, kind="cubic", fill_value="extrapolate")
    interpolated_uv = uv_curve(output_time).transpose((1,2,0))

    z_batch = query["z"].cpu().numpy()
    z_batch = z_batch.transpose((2,0,1))
    z_curve = interpolate.interp1d(time_batch, z_batch, kind="cubic", fill_value="extrapolate")
    interpolated_z = z_curve(output_time).transpose((1,2,0))
    
    interpolated_query["pos"] = torch.tensor(interpolated_pos, dtype=torch.float)
    interpolated_query["grasp_state"] = torch.tensor(interpolated_grasp, dtype=torch.float)
    interpolated_query["uv"] = torch.tensor(interpolated_uv, dtype=torch.float)
    interpolated_query["z"] = torch.tensor(interpolated_z, dtype=torch.float)
    interpolated_query["time"] = repeat(torch.tensor(output_time), "T -> B T N", B=B, N=1)
    
    interpolated_rot = torch.tensor(np.array(interpolated_rot), dtype=torch.float)
    if rot_mode == "6d":
        interpolated_rot = pytorch3d.transforms.matrix_to_rotation_6d(interpolated_rot)
    interpolated_query["rotation"] = interpolated_rot
    return interpolated_query

def denorm_uv(uv, image_size):
    """
    Preprocess includes
    1. denormalize uv from [-1, 1] to [0, image_size]
    """
    if uv.dim() == 2:
        u, v = uv[:, 0], uv[:, 1]
    elif uv.dim() == 1:
        u, v = uv[0], uv[1]

    h, w = image_size

    denorm_u = (u + 1) / 2 * (w - 1)
    denorm_v = (v + 1) / 2 * (h - 1)

    denorm_uv = torch.stack([denorm_u, denorm_v], dim=(uv.dim()-1))
    return denorm_uv

def get_pos(query, intrinsic, image_size=(256,256)):
    uv = query["uv"]
    z = query["z"]
    h, w = image_size
    device = uv.device

    u, v = uv[:,:,0], uv[:,:,1]
    u = (u + 1) / 2 * (w - 1)
    v = (v + 1) / 2 * (h - 1)
    uv_denorm = torch.stack([u, v], 2)
    ones = torch.ones(*uv_denorm.shape[:-1], 1).to(device)
    uv_denorm = torch.cat([uv_denorm, ones], 2)
    intrinsic = torch.tensor(intrinsic, dtype=torch.float).to(device)
    inv_intrinsic = torch.linalg.inv(intrinsic)

    xy = torch.einsum('ij,bkj->bki', inv_intrinsic, uv_denorm)
    xyz = xy * z
    query["pos"] = xyz
    return query

def convert_rotation_6d_to_matrix(query_list):
    for i, query in enumerate(query_list):
        query_list[i]["rotation"] = pytorch3d.transforms.rotation_6d_to_matrix(query["rotation"])
    return query_list

def str2bool(s):
    return s.lower() in ('true', '1')

# evaluation

def calculate_dtw_pos(pred_action, gt_action):
    pred_xyz = np.array(pred_action)[:,:3] * 1000
    gt_xyz = np.array(gt_action)[:,:3] * 1000

    print("calculate dtw pose")
    dtw_error_xyz, path_xyz = fastdtw(pred_xyz, gt_xyz, dist=euclidean)
    error_xyz_list = error_divide_time(pred_xyz, gt_xyz, euclidean, path_xyz)
    mean_dtw_xyz = dtw_error_xyz / len(path_xyz)

    dtw_error_x, path_x = fastdtw(pred_xyz[:,0:1], gt_xyz[:,0:1], dist=euclidean)
    error_x_list = error_divide_time(pred_xyz[:,0:1], gt_xyz[:,0:1], euclidean, path_x)
    mean_dtw_x = dtw_error_x / len(path_x)

    dtw_error_y, path_y = fastdtw(pred_xyz[:,1:2], gt_xyz[:,1:2], dist=euclidean)
    error_y_list = error_divide_time(pred_xyz[:,1:2], gt_xyz[:,1:2], euclidean, path_y)
    mean_dtw_y = dtw_error_y / len(path_y)

    dtw_error_z, path_z = fastdtw(pred_xyz[:,2:3], gt_xyz[:,2:3], dist=euclidean)
    error_z_list = error_divide_time(pred_xyz[:,2:3], gt_xyz[:,2:3], euclidean, path_z)
    mean_dtw_z = dtw_error_z / len(path_z)

    return mean_dtw_xyz, mean_dtw_x, mean_dtw_y, mean_dtw_z, error_xyz_list, error_x_list, error_y_list, error_z_list

def calculate_dtw_angle(pred_action, gt_action):
    pred_quat = np.array(pred_action)[:,3:7]
    gt_quat = np.array(gt_action)[:,3:7]

    r = R.from_quat(pred_quat)
    pred_eular = r.as_euler('xyz')

    r = R.from_quat(gt_quat)
    gt_eular = r.as_euler('xyz')

    def angle_euclidean(angle1, angle2):
        diff_eular = angle1 - angle2
        diff_eular = np.where(abs(diff_eular) > np.pi, (2*np.pi) - abs(diff_eular), abs(diff_eular))
        return np.linalg.norm(diff_eular)

    print("calculate dtw angle")
    dtw_error_xyz, path_xyz = fastdtw(pred_eular, gt_eular, dist=angle_euclidean)
    error_xyz_list = error_divide_time(pred_eular, gt_eular, angle_euclidean, path_xyz)
    mean_dtw_xyz = dtw_error_xyz / len(path_xyz)

    dtw_error_x, path_x = fastdtw(pred_eular[:,0], gt_eular[:,0], dist=angle_euclidean)
    error_x_list = error_divide_time(pred_eular[:,0], gt_eular[:,0], angle_euclidean, path_x)
    mean_dtw_x = dtw_error_x / len(path_x)

    dtw_error_y, path_y = fastdtw(pred_eular[:,1], gt_eular[:,1], dist=angle_euclidean)
    error_y_list = error_divide_time(pred_eular[:,1], gt_eular[:,1], angle_euclidean, path_y)
    mean_dtw_y = dtw_error_y / len(path_y)

    dtw_error_z, path_z = fastdtw(pred_eular[:,2], gt_eular[:,2], dist=angle_euclidean)
    error_z_list = error_divide_time(pred_eular[:,2], gt_eular[:,2], angle_euclidean, path_z)
    mean_dtw_z = dtw_error_z / len(path_z)

    return mean_dtw_xyz, mean_dtw_x, mean_dtw_y, mean_dtw_z, error_xyz_list, error_x_list, error_y_list, error_z_list

def calculate_euclid_pos(pred_action, gt_action):
    pred_xyz = np.array(pred_action)[:,:3] * 1000
    # pred_xyz = np.nan_to_num(pred_xyz, nan=1e+8)
    gt_xyz = np.array(gt_action)[:,:3] * 1000

    print("calculate euclid pose")
    error_list = list(map(euclidean, pred_xyz, gt_xyz))
    mean_error = sum(error_list) / len(gt_xyz)

    error_list_x = list(map(euclidean, pred_xyz[:,:1], gt_xyz[:,:1]))
    mean_error_x = sum(error_list_x) / len(gt_xyz)

    error_list_y = list(map(euclidean, pred_xyz[:,1:2], gt_xyz[:,1:2]))
    mean_error_y = sum(error_list_y) / len(gt_xyz)

    error_list_z = list(map(euclidean, pred_xyz[:,2:3], gt_xyz[:,2:3]))
    mean_error_z = sum(error_list_z) / len(gt_xyz)

    return mean_error, mean_error_x, mean_error_y, mean_error_z, error_list, error_list_x, error_list_y, error_list_z

def calculate_euclid_angle(pred_action, gt_action):
    pred_quat = np.array(pred_action)[:,3:7]
    gt_quat = np.array(gt_action)[:,3:7]

    r = R.from_quat(pred_quat)
    pred_eular = r.as_euler('xyz')

    r = R.from_quat(gt_quat)
    gt_eular = r.as_euler('xyz')

    def angle_euclidean(angle1, angle2):
        diff_eular = angle1 - angle2
        diff_eular = np.where(abs(diff_eular) > np.pi, (2*np.pi) - abs(diff_eular), abs(diff_eular))
        return np.linalg.norm(diff_eular)

    print("calculate euclid angle")
    error_list = list(map(angle_euclidean, pred_eular, gt_eular))
    mean_error = sum(error_list) / len(gt_eular)

    error_list_x = list(map(angle_euclidean, pred_eular[:,0], gt_eular[:,0]))
    mean_error_x = sum(error_list_x) / len(gt_eular)

    error_list_y = list(map(angle_euclidean, pred_eular[:,1], gt_eular[:,1]))
    mean_error_y = sum(error_list_y) / len(gt_eular)

    error_list_z = list(map(angle_euclidean, pred_eular[:,2], gt_eular[:,2]))
    mean_error_z = sum(error_list_z) / len(gt_eular)

    return mean_error, mean_error_x, mean_error_y, mean_error_z, error_list, error_list_x, error_list_y, error_list_z

def calculate_euclid_grasp(pred_action, gt_action):
    pred_grasp = np.array(pred_action)[:,7:8]
    gt_grasp = np.array(gt_action)[:,7:8]

    print("calculate euclid grasp")
    error_list = list(map(euclidean, pred_grasp, gt_grasp))
    mean_error = sum(error_list) / len(gt_grasp)

    return mean_error, error_list

def error_divide_time(pred, gt, dist, path):
    # error_list = []
    # for i,j in path:
    #     error = dist(pred[i], gt[j])
    #     error_list.append(error)
    
    error_list = [0] * len(gt)
    for i,j in path:
        error = dist(pred[i],gt[j])
        error_list[j] = error
    return error_list

def output2action(query, obs):
    world2camera_matrix = torch.tensor(obs.misc["front_camera_extrinsics"], dtype=torch.float)
    gripper_rotation = query['rotation'][0].cpu().detach()
    gripper_pos = torch.unsqueeze(query['pos'][0].cpu().detach(), 2)
    B, _, _ = gripper_pos.shape
    temp = torch.tensor([0.,0.,0.,1.])
    temp = repeat(temp, 'N -> B D N', B=B, D=1)
    world2camera_matrix = repeat(world2camera_matrix, "N D -> B N D", B=B)

    gripper_matrix = torch.cat([gripper_rotation, gripper_pos], 2)
    gripper_matrix = torch.cat([gripper_matrix, temp], 1)
    gripper_matrix = torch.einsum('bij,bjk->bik', world2camera_matrix, gripper_matrix)

    action_list = []
    for i in range(len(gripper_matrix)):
        r = R.from_matrix(gripper_matrix[i,:3,:3])        
        quat = r.as_quat()
        gripper_action = np.append(gripper_matrix[i,:3,3], quat)

        grasp = query["grasp_state"][0,i].numpy()
        # grasp =  np.round(grasp)
        gripper_action = np.append(gripper_action, grasp)
        action_list.append(gripper_action)
    return action_list

def check_img(img, obs):
    if img.dim() == 4:
        img = img[0,:3]
    elif img.dim() == 3:
        img = img[:3]
    else:
        raise ValueError("TODO")
        
    img1 = np.array(torchvision.transforms.ToPILImage()(img))
    img2 = obs.front_rgb
    
    diff = np.mean(np.abs((img1.astype(np.int16) - img2.astype(np.int16))))
    if diff > 3.:
        print(f"check img. diff: {diff}")
        print("recommend you to re-create the dataset")
        return True
    else:
        return False

def get_gt_pose(base_dir):
    pickle_list = os.listdir(base_dir)
    pickle_list.sort()
    gt_state_list = []
    gt_matrix_list = []
    gt_image_path_list = []
    for pickle_index, pickle_name in enumerate(pickle_list):
        pickle_path = os.path.join(base_dir, pickle_name)
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        gt_state_list.append(np.append(data["gripper_pose"], data["gripper_open"]))
        gt_matrix_list.append(data["gripper_matrix"])
        
    return gt_state_list, gt_matrix_list

def make_video(pil_list, file_path, size, success, fps=20,):
    videodims = size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
    video = cv2.VideoWriter(file_path, fourcc, fps, videodims, True)
    #draw stuff that goes on every frame here
    for index, pil_img in enumerate(pil_list):
        imtemp = pil_img.copy()
        image_editable = ImageDraw.Draw(imtemp)
        if success:
            judge = "Success"
            color = (237, 230, 211)
        else:
            judge = "Fail"
            color = (255, 0, 0)
        image_editable.text((15,15), 'index:{}\n judge: {}'.format(index,judge), color)
        # draw frame specific stuff here.
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

def make_video_verbose(pil_list, file_path, size, success, pose_error_list, angle_error_list, grasp_error_list, fps=20,):
    videodims = size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
    video = cv2.VideoWriter(file_path, fourcc, fps, videodims, True)
    #draw stuff that goes on every frame here
    for index, pil_img in enumerate(pil_list):
        imtemp = pil_img.copy()
        image_editable = ImageDraw.Draw(imtemp)
        if success:
            judge = "Success"
            color = (237, 230, 211)
        else:
            judge = "Fail"
            color = (255, 0, 0)
        
        image_editable.text((15,15), 'index:{}\n judge: {}'.format(index,judge), color)
        # draw frame specific stuff here.
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

class Concat_query_keys():
    
    def __init__(self, target_key=['rotation','grasp_state','uv','z'], target_dim=[6,1,2,1]):
        self.target_key = target_key
        self.target_dim = target_dim
        
    def concat_query(self, query):
        return torch.cat([query[key] for key in self.target_key], 2)
        
    def split_query(self, query):
        """
        query -> torch.tensor: (Batch, Time, Dim)
        """
        splited_query = {}
        sum_dim = 0
        for key, dim in zip(self.target_key, self.target_dim):
            splited_query[key] = query[:,:,sum_dim:sum_dim+dim]
            sum_dim += dim
        return splited_query