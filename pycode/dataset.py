import os
import random
import pickle
import copy
import time
import math

import torch
import torchvision

import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
from PIL import Image
from einops import repeat, rearrange
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from scipy import interpolate

from .misc import get_pos

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    This code is copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

class RLBench_DMOEBM(torch.utils.data.Dataset):
    """
    RLBench dataset for train IBC model
    
    Attributes
    ----------
    index_list: list[int]
        List of valid index of dictionaly.
    """
    
    def __init__(self, mode, cfg, save_dataset=False, debug=False, num_frame=100,
                rot_mode="6d", img_aug=True, keys=["pos","uv","z","rotation","grasp_state","time"], aug_type="none"):

        # set dataset root
        if cfg.DATASET.NAME == "RLBench":
            self.data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)
        else:
            raise ValueError("Invalid dataset name")

        self.cfg = cfg
        self.num_frame = num_frame
        self.rot_mode = rot_mode
        self.seed = 0
        self.info_dict = {}
        self.mode = mode
        random.seed(self.seed)

        task_names = cfg.DATASET.RLBENCH.TASK_NAME
        print(f"TASK: {task_names}")
        self._pickle_file_name = '{}_{}_{}.pickle'.format(cfg.DATASET.NAME,mode,task_names)
        self._pickle_path = os.path.join(self.data_root_dir, 'pickle', self._pickle_file_name)
        if not os.path.exists(self._pickle_path) or save_dataset:
            # create dataset
            print('There is no pickle data')
            print('create pickle data')
            self.add_data(self.data_root_dir, cfg)
            self.preprocess()
            print('done')
            
            # save json data
            print('save pickle data')
            os.makedirs(os.path.join(self.data_root_dir, 'pickle'), exist_ok=True)
            with open(self._pickle_path, mode='wb') as f:
                pickle.dump(self.info_dict,f)
            print('done')
        else:
            # load json data
            print('load pickle data')
            with open(self._pickle_path, mode='rb') as f:
                self.info_dict = pickle.load(f)
            print('done')

        self.ToTensor = torchvision.transforms.ToTensor()

        self.debug = debug
        self.without_img = False
        self.aug_flag = img_aug
        if mode == "train":
            self.img_aug = iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=0.05*255),
                        iaa.JpegCompression(compression=(30, 70)),
                        iaa.WithBrightnessChannels(iaa.Add((-40, 40))),
                        iaa.AverageBlur(k=(2, 5)),
                        iaa.CoarseDropout(0.02, size_percent=0.5),
                        iaa.Identity()
                    ])
        elif mode == "val" or mode == "test":
            if aug_type == "noise":
                self.img_aug = iaa.AdditiveGaussianNoise(scale=0.05*255)
            elif aug_type == "jpeg":
                self.img_aug = iaa.JpegCompression(compression=(30, 90))
            elif aug_type == "blur":
                self.img_aug = iaa.AverageBlur(k=(5, 8))
            elif aug_type == "none":
                self.img_aug = iaa.Identity()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        self.key_list = keys

    def __len__(self):
        return len(self.info_dict["sequence_index_list"])

    def __getitem__(self, data_index):
        # get image
        start_index, end_index = self.info_dict["sequence_index_list"][data_index]
        
        if self.without_img == False:
            rgb_path = os.path.join(self.data_root_dir, self.info_dict["data_list"][start_index]['image_dir'], "front_rgb_00000000.png")
            rgb_image = Image.open(rgb_path)
            if self.aug_flag:
                rgb_image = np.array(rgb_image)
                rgb_image = Image.fromarray(self.img_aug(image=rgb_image))
            rgb_image = self.ToTensor(rgb_image)
        
            depth_path = os.path.join(self.data_root_dir, self.info_dict["data_list"][start_index]['image_dir'], "front_depth_00000000.pickle")
            with open(depth_path, 'rb') as f:
                depth_image = pickle.load(f)
            depth_image = torch.unsqueeze(torch.tensor(np.array(depth_image), dtype=torch.float), 0)
            
            image = torch.cat([rgb_image, depth_image], 0)
        else:
            image = torch.zeros(1)
        
        sequence_index_list = [i / self.num_frame for i in range(self.num_frame + 1)]
        
        pos = self.info_dict["pos_curve_list"][data_index](sequence_index_list).transpose((1,0))
        
        if self.rot_mode == "quat":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_quat()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "euler":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_euler('zxy', degrees=True)
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "matrix":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_matrix()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "6d":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_matrix()
            rot = matrix_to_rotation_6d(torch.tensor(rot, dtype=torch.float))
        else:
            raise ValueError("invalid mode for get_gripper")
        
        grasp = self.info_dict["grasp_state_curve_list"][data_index](sequence_index_list).transpose((1,0))
        uv = self.info_dict["uv_curve_list"][data_index](sequence_index_list).transpose((1,0))
        z = self.info_dict["z_curve_list"][data_index](sequence_index_list).transpose((1,0))
        
        action_dict = {}
        if "pos" in self.key_list:
            action_dict["pos"] = torch.tensor(pos, dtype=torch.float)
        if "rotation" in self.key_list:
            action_dict["rotation"] = rot
        if "grasp_state" in self.key_list:
            action_dict["grasp_state"] = torch.tensor(grasp, dtype=torch.float)
        if "uv" in self.key_list:
            action_dict["uv"] = torch.tensor(uv, dtype=torch.float)
        if "z" in self.key_list:
            action_dict["z"] = torch.tensor(z, dtype=torch.float)
        if "time" in self.key_list:
            action_dict["time"] = torch.unsqueeze(torch.tensor(sequence_index_list, dtype=torch.float), 1)

        return image, action_dict
    
    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list of data_dict
        data_dict = {
        'filename': str -> name of each data except file name extension. e.g. 00000
        'image_dir': str -> path to image dir which includes rgb and depth images
        'pickle_dir': str -> path to pickle dir.
        'end_index': index of data when task will finish
        'start_index': index of date when task started
        'gripper_state_change': index of gripper state is changed. The value is 0 when the gripper state is not changed
        }
        next_index_list: If we set self.next_len > 1, next frame is set to current_index + self.next_len. However, if the next frame index skip the grassping frame, it is modified to the grassping frame.
        index_list: this is used for index of __getitem__()
        sequence_index_list: this list contains lists of [start_index of sequennce, end_index of sequence]. so sequence_index_list[0] returns start and end frame index of 0-th sequence.
        """
        # for data preparation
        self.info_dict["data_list"] = []
        self.info_dict["index_list"] = []
        self.info_dict["sequence_index_list"] = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        task_name = cfg.DATASET.RLBENCH.TASK_NAME

        print(f"taskname: {task_name}")
        task_path = os.path.join(folder_path, task_name)
        
        sequence_list = os.listdir(task_path)
        sequence_list.sort()
        
        for sequence_index in tqdm(sequence_list):
            start_index = index
            image_folder_path = os.path.join(task_name, sequence_index, 'image')
            pickle_folder_path = os.path.join(task_name, sequence_index, 'base_data')
            pickle_name_list = os.listdir(os.path.join(folder_path, pickle_folder_path))
            pickle_name_list.sort()
            end_index = start_index + len(pickle_name_list) - 1

            past_gripper_open = 1.0 # default gripper state is open. If not, please change
            pickle_data_list = []
            for pickle_name in pickle_name_list:
                # gripper state check to keep grasping frame
                with open(os.path.join(folder_path, pickle_folder_path, pickle_name), 'rb') as f:
                    pickle_data = pickle.load(f)
                    pickle_data_list.append(pickle_data)
                
                head, ext = os.path.splitext(pickle_name)
                data_dict = {}
                data_dict['image_dir'] = image_folder_path
                data_dict['filename'] = os.path.join(head)
                data_dict['pickle_path'] = pickle_folder_path
                data_dict['start_index'] = start_index
                data_dict['end_index'] = end_index
                self.info_dict["data_list"].append(data_dict)
                # get camera info
                camera_intrinsic = pickle_data['front_intrinsic_matrix']
                data_dict["camera_intrinsic"] = camera_intrinsic
                gripper_open = pickle_data['gripper_open']
                data_dict['gripper_state'] = gripper_open
                
            # image size
            if index == 0:
                rgb_path = os.path.join(folder_path, data_dict['image_dir'], "front_rgb_{}.png".format(head))
                rgb_image = Image.open(rgb_path)
                image_size = rgb_image.size
            self.info_dict["image_size"] = image_size

            # get gripper info
            pose, rotation = self.get_gripper(pickle_data_list)

            # get uv cordinate and pose image
            uv = self.get_uv(pose, camera_intrinsic)
            uv = self.preprocess_uv(uv, image_size)
            
            for j in range(len(pose)):
                self.info_dict["data_list"][index]["pose"] = pose[j]
                self.info_dict["data_list"][index]["rotation"] = rotation[j]
                self.info_dict["data_list"][index]["uv"] = uv[j]
                self.info_dict["data_list"][index]['current_index'] = index
                index += 1
        
            self.info_dict["sequence_index_list"].append([start_index, end_index])
            
    def preprocess(self):
        print("start preprocess")
        self.info_dict["max_len"] = 0
        self.info_dict["pos_curve_list"] = []
        self.info_dict["rot_curve_list"] = []
        self.info_dict["grasp_state_curve_list"] = []
        self.info_dict["uv_curve_list"] = []
        self.info_dict["z_curve_list"] = []


        for i, (start_index, end_index) in enumerate(self.info_dict["sequence_index_list"]):
            print(f"{i}/{len(self.info_dict['sequence_index_list'])}")

            if self.info_dict["max_len"] < end_index - start_index + 1:
                self.info_dict["max_len"] = end_index - start_index + 1

            index_list = [index for index in range(start_index, end_index+1)]
            time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch = self.get_list(index_list, start_index, end_index)
            pos_curve, rot_curve, grasp_curve, uv_curve, z_curve = self.get_spline_curve(time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch)
            
            self.info_dict["pos_curve_list"].append(pos_curve)
            self.info_dict["rot_curve_list"].append(rot_curve)
            self.info_dict["grasp_state_curve_list"].append(grasp_curve)
            self.info_dict["uv_curve_list"].append(uv_curve)
            self.info_dict["z_curve_list"].append(z_curve)


    def preprocess_uv(self, uv, image_size):
        """
        Preprocess includes
        1. convert to torch.tensor
        2. convert none to 0.
        3. normalize uv from [0, image_size] to [-1, 1]
        """
        u, v = uv[:, 0], uv[:, 1]
        h, w = image_size
        u = (u / (w - 1) * 2) - 1
        v = (v / (h - 1) * 2) - 1
        uv = np.stack([u, v], 1)
        return uv
    
    def postprocess_uv(self, uv, image_size):
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
        
    def get_gripper(self, pickle_list):
        gripper_pos_WorldCor = np.array([np.append(pickle_data['gripper_pose'][:3], 1) for pickle_data in pickle_list])
        gripper_matrix_WorldCor = np.array([pickle_data['gripper_matrix'] for pickle_data in pickle_list])
        # gripper_open = torch.unsqueeze(gripper_open, 1)

        world2camera_matrix = np.array([pickle_data['front_extrinsic_matrix'] for pickle_data in pickle_list])
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.einsum('bij,bj->bi', camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.einsum('bij,bjk->bik', camera2world_matrix, gripper_matrix_WorldCor)
        gripper_rot_CamCor = R.from_matrix(gripper_matrix_CamCor[:,:3,:3])
            
        # return torch.tensor(gripper_pose_CamCor[:, :3], dtype=torch.float), torch.tensor(gripper_rot_CamCor, dtype=torch.float)
        return gripper_pose_CamCor[:, :3], gripper_rot_CamCor

    def update_seed(self):
        # change seed. augumentation will be changed.
        self.seed += 1
        random.seed(self.seed)
        
    @staticmethod
    def get_task_names(task_list):
        for i, task in enumerate(task_list):
            if i == 0:
                task_name = task
            else:
                task_name = task_name + "_" + task
        return task_name

    def get_uv(self, pos_data, intrinsic_matrix):
        # transfer position data(based on motive coordinate) to camera coordinate
        B, _ = pos_data.shape
        z = np.repeat(pos_data[:, 2], 3).reshape((B,3))
        pos_data = pos_data / z # u,v,1
        uv_result = np.einsum('ij,bj->bi', intrinsic_matrix, pos_data)
        return uv_result[:, :2]
    
    def get_list(self, index_list, start_index, end_index):
        pose_batch = []
        rotation_batch = []
        grasp_state_batch = []
        uv_batch = []
        z_batch = []
        time_batch = []

        # get pickle data
        start = time.time()
        for i,index in enumerate(index_list):
            data_dict = self.info_dict["data_list"][index]
            pose_batch.append(data_dict["pose"])
            uv_batch.append(data_dict["uv"])
            z_batch.append(data_dict["pose"][2:])
            grasp_state_batch.append([data_dict['gripper_state']])

            gripper_rot_CamCor = data_dict["rotation"]
            gripper_rot_CamCor = gripper_rot_CamCor.as_matrix()

            rotation_batch.append(gripper_rot_CamCor)

            normalized_time = (index - start_index) / (end_index - start_index)
            time_batch.append(normalized_time)
        
        time_batch = np.array(time_batch)
        pose_batch = np.array(pose_batch)
        rotation_batch = np.array(rotation_batch)
        grasp_state_batch = np.array(grasp_state_batch)
        uv_batch = np.array(uv_batch)
        z_batch = np.array(z_batch)
        return time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch
        
    def get_spline_curve(self, time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch):
        pose_batch = pose_batch.transpose((1,0))
        pos_curve = interpolate.interp1d(time_batch, pose_batch, kind="cubic", fill_value="extrapolate")
        # interpolated_pos = pos_curve(output_time).transpose((1,0))

        
        query_rot = R.from_matrix(rotation_batch)
        rot_curve = RotationSpline(time_batch, query_rot)
        # interpolated_rot = spline(output_time).as_matrix()

        grasp_state_batch = grasp_state_batch.transpose((1,0))
        grasp_curve = interpolate.interp1d(time_batch, grasp_state_batch, fill_value="extrapolate")

        
        uv_batch = uv_batch.transpose((1,0))
        uv_curve = interpolate.interp1d(time_batch, uv_batch, kind="cubic", fill_value="extrapolate")

        z_batch = z_batch.transpose((1,0))
        z_curve = interpolate.interp1d(time_batch, z_batch, kind="cubic", fill_value="extrapolate")
        
        return pos_curve, rot_curve, grasp_curve, uv_curve, z_curve

class RLBench_Retrieval(RLBench_DMOEBM):

    def __init__(self, mode, cfg, save_dataset=False, debug=False, num_frame=100,
                rot_mode="6d", img_aug=True, keys=["pos","uv","z","rotation","grasp_state","time"], r_weight=0.1, temperature=1.0, rank=3):

        # set dataset root
        if cfg.DATASET.NAME == "RLBench":
            self.data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)
        else:
            raise ValueError("Invalid dataset name")
        
        self.cfg = cfg
        self.debug = debug
        self.num_frame = num_frame
        self.rot_mode = rot_mode
        self.key_list = keys
        self.seed = 0
        self.mode = mode
        random.seed(self.seed)

        self.ToTensor = torchvision.transforms.ToTensor()

        self.without_img = False
        self.aug_flag = img_aug
        self.img_aug = iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=0.05*255),
                        iaa.JpegCompression(compression=(30, 70)),
                        iaa.WithBrightnessChannels(iaa.Add((-40, 40))),
                        iaa.AverageBlur(k=(2, 5)),
                        iaa.CoarseDropout(0.02, size_percent=0.5),
                        iaa.Identity()
                    ])

        # retrieval config
        self.rank = rank
        self.rotation_weight_for_retrieval = r_weight
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=0)

        self.info_dict = {}
        task_names = cfg.DATASET.RLBENCH.TASK_NAME
        print(f"TASK: {task_names}")
        self._pickle_file_name = '{}_{}_{}_Retrieval.pickle'.format(cfg.DATASET.NAME,mode,task_names)
        self._pickle_path = os.path.join(self.data_root_dir, 'pickle', self._pickle_file_name)
        if not os.path.exists(self._pickle_path) or save_dataset:
            # create dataset
            print('There is no pickle data')
            print('create pickle data')
            self.add_data(self.data_root_dir, cfg)
            self.preprocess()
            vecs = self.get_all_vec()
            sorted_distances, sorted_indices = self.setup_retrieval(vecs)
            self.info_dict["sorted_distances"] = sorted_distances[:,1:]
            self.info_dict["sorted_indices"] = sorted_indices[:,1:]
            print('done')
            
            # save json data
            print('save pickle data')
            os.makedirs(os.path.join(self.data_root_dir, 'pickle'), exist_ok=True)
            with open(self._pickle_path, mode='wb') as f:
                pickle.dump(self.info_dict,f)
            print('done')
        else:
            # load json data
            print('load pickle data')
            with open(self._pickle_path, mode='rb') as f:
                self.info_dict = pickle.load(f)
            print('done')
        self.vecs = None
        self.all_query = None

    def __getitem__(self, index):
        """
        Return:
        target_image (torch.tensor: B C H W)
        target_motion (dict): Components of this dictionary are determined by self.keys
            key: "pos", value: torch.tensor B(batch) S(sequence length) 3(dim); position of robot hand in camera coordinate system
            key: "uv", value: torch.tensor B(batch) S(sequence length) 2(dim); position of robot hand in image coordinate system 
            key: "z", value: torch.tensor B S 1; depth of robot hand in camera coordinate system
            key: "rotation", value: torch.tensor B S D(depends on representation. 6d->6)
            key: "grasp_state", value: torch.tensor B S 1
        positive_image (or motion); positive means this image (or motion) is similar to target image (or motion).
        negative_image (or motion); negative means this image (or motion) is far from target image (or motion).
        """
        target_index = index
        target_image = self.get_image_from_index(target_index)
        target_motion = self.get_motion_from_index(target_index)

        positive_index = self.info_dict["sorted_indices"][target_index, random.randint(0, self.rank-1)]
        positive_image = self.get_image_from_index(positive_index)
        positive_motion = self.get_motion_from_index(positive_index)

        negative_index_prob = self.softmax(-self.info_dict["sorted_distances"][target_index, self.rank:] / self.temperature)
        distribution = torch.distributions.Categorical(negative_index_prob)
        negative_index = distribution.sample()
        negative_image = self.get_image_from_index(negative_index)
        negative_motion = self.get_motion_from_index(negative_index)

        return target_image, target_motion, positive_image, positive_motion, negative_image, negative_motion

    def retrieve_k_sample(self, target_query, k=8):
        # set database
        if self.all_query == None:
            self.vecs = self.get_all_vec()
            self.prepare_database_for_retrieval()

        # compute vec to get kNN sample
        if "pos" not in target_query.keys():
            target_query = get_pos(target_query, self.info_dict["data_list"][0]["camera_intrinsic"])
        target_vecs = self.get_vec_from_query(target_query)
        device = target_vecs.device
        B, _ = target_vecs.shape

        # compute distances
        distances = torch.cdist(target_vecs, self.vecs.to(device))
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        near_queries = {}
        for key in self.all_query.keys():
            _, S, D = self.all_query[key].shape
            index_nears = repeat(sorted_indices[:, k-1], "B -> B S D",S=S,D=D)
            near_queries[key] = torch.gather(self.all_query[key].to(device), 0, index_nears)

        imgs = [self.get_image_from_index(sorted_indices[index, k-1]) for index in range(B)]
        imgs = torch.stack(imgs, 0)
        return imgs, near_queries

    def get_all_vec(self):
        print("loading dataset")
        all_query = {}
        temp_key = copy.deepcopy(self.key_list)
        self.key_list = ["pos", "rotation"]
        for i in tqdm(range(self.__len__())):
            query = self.get_motion_from_index(i)
            if i == 0:
                for key in query.keys():
                    all_query[key] = torch.unsqueeze(query[key], 0)
            else:
                for key in query.keys():
                    all_query[key] = torch.cat([all_query[key], torch.unsqueeze(query[key], 0)], 0)
                    
        all_vec = self.get_vec_from_query(all_query)
        self.key_list = temp_key
        return all_vec
    
    def prepare_database_for_retrieval(self):
        print("prepare database for retrieval")
        all_query = {}
        for i in tqdm(range(self.__len__())):
            query = self.get_motion_from_index(i)
            if i == 0:
                for key in query.keys():
                    all_query[key] = torch.unsqueeze(query[key], 0)
            else:
                for key in query.keys():
                    all_query[key] = torch.cat([all_query[key], torch.unsqueeze(query[key], 0)], 0)
        self.all_query = all_query

    def setup_retrieval(self, vecs):
        distances = torch.cdist(vecs, vecs)
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        return sorted_distances, sorted_indices

    def get_vec_from_query(self, query):
        """
        query: dict
        query["???"]: torch.array, shape:(Sequence_Length, Dim of ???), e.g, shape of query["pos"] = (101, 3)
        """
        pos = rearrange(query["pos"], "B S D -> B (S D)")
        rot = rearrange(query["rotation"], "B S D -> B (S D)") * self.rotation_weight_for_retrieval
        return torch.cat([pos, rot], 1)

    def get_motion_from_index(self, data_index):
        # get image
        start_index, end_index = self.info_dict["sequence_index_list"][data_index]
        
        sequence_index_list = [i / self.num_frame for i in range(self.num_frame + 1)]
        
        pos = self.info_dict["pos_curve_list"][data_index](sequence_index_list).transpose((1,0))
        
        if self.rot_mode == "quat":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_quat()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "euler":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_euler('zxy', degrees=True)
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "matrix":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_matrix()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "6d":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_matrix()
            rot = matrix_to_rotation_6d(torch.tensor(rot, dtype=torch.float))
        else:
            raise ValueError("invalid mode for get_gripper")
        
        grasp = self.info_dict["grasp_state_curve_list"][data_index](sequence_index_list).transpose((1,0))
        uv = self.info_dict["uv_curve_list"][data_index](sequence_index_list).transpose((1,0))
        z = self.info_dict["z_curve_list"][data_index](sequence_index_list).transpose((1,0))
        
        action_dict = {}
        if "pos" in self.key_list:
            action_dict["pos"] = torch.tensor(pos, dtype=torch.float)
        if "rotation" in self.key_list:
            action_dict["rotation"] = rot
        if "grasp_state" in self.key_list:
            action_dict["grasp_state"] = torch.tensor(grasp, dtype=torch.float)
        if "uv" in self.key_list:
            action_dict["uv"] = torch.tensor(uv, dtype=torch.float)
        if "z" in self.key_list:
            action_dict["z"] = torch.tensor(z, dtype=torch.float)
        if "time" in self.key_list:
            action_dict["time"] = torch.unsqueeze(torch.tensor(sequence_index_list, dtype=torch.float), 1)

        return action_dict

    def get_image_from_index(self, index):
        # get image
        start_index, end_index = self.info_dict["sequence_index_list"][index]
        
        if self.without_img == False:
            rgb_path = os.path.join(self.data_root_dir, self.info_dict["data_list"][start_index]['image_dir'], "front_rgb_00000000.png")
            rgb_image = Image.open(rgb_path)
            if (self.mode == "train") and self.aug_flag:
                rgb_image = np.array(rgb_image)
                rgb_image = Image.fromarray(self.img_aug(image=rgb_image))
            rgb_image = self.ToTensor(rgb_image)
        
            depth_path = os.path.join(self.data_root_dir, self.info_dict["data_list"][start_index]['image_dir'], "front_depth_00000000.pickle")
            with open(depth_path, 'rb') as f:
                depth_image = pickle.load(f)
            depth_image = torch.unsqueeze(torch.tensor(np.array(depth_image), dtype=torch.float), 0)
            
            image = torch.cat([rgb_image, depth_image], 0)
        else:
            image = torch.zeros(1)
        
        return image

class RLBench_BYOL_ALLImage(torch.utils.data.Dataset):
    """
    RLBench dataset for train IBC model
    
    Attributes
    ----------
    index_list: list[int]
        List of valid index of dictionaly.
    """
    
    def __init__(self, mode, cfg, save_dataset=False, debug=False, num_frame=100, rot_mode="quat"):

        # set dataset root
        if cfg.DATASET.NAME == "RLBench":
            data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)
        else:
            raise ValueError("Invalid dataset name")

        self.cfg = cfg
        self.num_frame = num_frame
        self.rot_mode = rot_mode
        self.seed = 0
        self.info_dict = {}
        self.mode = mode
        random.seed(self.seed)

        task_names = cfg.DATASET.RLBENCH.TASK_NAME
        print(f"TASK: {task_names}")
        self.add_data(data_root_dir, cfg)
        self.ToTensor = torchvision.transforms.ToTensor()

        self.debug = debug

    def __len__(self):
        return len(self.info_dict["data_list"])

    def __getitem__(self, data_index):
        # get image
        data_dict = self.info_dict["data_list"][data_index]
        
        rgb_path = data_dict["image_path"]
        rgb_image = Image.open(rgb_path)
        rgb_image = self.ToTensor(rgb_image)
    
        # get pickle data
        # pickle_path = data_dict['pickle_path']
        # with open(pickle_path, 'rb') as f:
            # pickle_data = pickle.load(f)

        # gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
        
        return rgb_image
    
    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list of data_dict
        data_dict = {
        'filename': str -> name of each data except file name extension. e.g. 00000
        'image_dir': str -> path to image dir which includes rgb and depth images
        'pickle_dir': str -> path to pickle dir.
        'end_index': index of data when task will finish
        'start_index': index of date when task started
        'gripper_state_change': index of gripper state is changed. The value is 0 when the gripper state is not changed
        }
        next_index_list: If we set self.next_len > 1, next frame is set to current_index + self.next_len. However, if the next frame index skip the grassping frame, it is modified to the grassping frame.
        index_list: this is used for index of __getitem__()
        sequence_index_list: this list contains lists of [start_index of sequennce, end_index of sequence]. so sequence_index_list[0] returns start and end frame index of 0-th sequence.
        """
        # for data preparation
        self.info_dict["data_list"] = []
        self.info_dict["index_list"] = []
        self.info_dict["sequence_index_list"] = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        task_name = cfg.DATASET.RLBENCH.TASK_NAME

        print(f"taskname: {task_name}")
        task_path = os.path.join(folder_path, task_name)
        
        sequence_list = os.listdir(task_path)
        sequence_list.sort()
        
        for sequence_index in tqdm(sequence_list):
            start_index = index
            image_folder_path = os.path.join(task_path, sequence_index, 'image')
            pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
            pickle_name_list = os.listdir(pickle_folder_path)
            pickle_name_list.sort()
            end_index = start_index + len(pickle_name_list) - 1

            past_gripper_open = 1.0 # default gripper state is open. If not, please change
            # pickle_data_list = []
            for pickle_name in pickle_name_list:
                # gripper state check to keep grasping frame
                # with open(os.path.join(pickle_folder_path, pickle_name), 'rb') as f:
                #     pickle_data = pickle.load(f)
                #     pickle_data_list.append(pickle_data)
                
                head, ext = os.path.splitext(pickle_name)
                image_path = os.path.join(image_folder_path, f"front_rgb_{head}.png")
                data_dict = {}
                data_dict['image_path'] = image_path
                data_dict['filename'] = os.path.join(head)
                data_dict['start_index'] = start_index
                data_dict['end_index'] = end_index
                # get camera info
                # camera_intrinsic = pickle_data['front_intrinsic_matrix']
                # data_dict["camera_intrinsic"] = camera_intrinsic
                self.info_dict["data_list"].append(data_dict)
                
            # image size
            if index == 0:
                rgb_image = Image.open(image_path)
                image_size = rgb_image.size
            self.info_dict["image_size"] = image_size

            self.info_dict["sequence_index_list"].append([start_index, end_index])

    def update_seed(self):
        # change seed. augumentation will be changed.
        self.seed += 1
        random.seed(self.seed)
        
    @staticmethod
    def get_task_names(task_list):
        for i, task in enumerate(task_list):
            if i == 0:
                task_name = task
            else:
                task_name = task_name + "_" + task
        return task_name