import torch
import torchvision

from tqdm import tqdm
from typing import Dict, Callable, List
from torchvision import models
from einops import rearrange, reduce, repeat
from sklearn.neighbors import NearestNeighbors

from .misc import load_checkpoint, get_pos

class Direct_Retrieval():
    
    def __init__(self, dataset, seq_len=101, n_neighbors=8):
        self.seq_len = seq_len
        self.all_vec, self.all_query = self.get_all_vec(dataset)
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors, metric=self.weighted_euclidian_distance)
        self.intrinsic = dataset.info_dict["data_list"][0]["camera_intrinsic"]
        
        print("fitting")
        self.neigh.fit(self.all_vec)
        
    def get_all_vec(self, dataset):
        print("loading dataset")
        all_query = {}
        for i in tqdm(range(len(dataset))):
            _, query = dataset[i]
            if i == 0:
                for key in query.keys():
                    all_query[key] = torch.unsqueeze(query[key], 0)
            else:
                for key in query.keys():
                    all_query[key] = torch.cat([all_query[key], torch.unsqueeze(query[key], 0)], 0)
            
        if "pos" not in query.keys():
            print("compute pos")
            all_query = get_pos(all_query, dataset.info_dict["data_list"][0]["camera_intrinsic"])
                    
        all_vec = self.get_vec_from_query(all_query)
        return all_vec, all_query
    
    def retrieve_k_sample(self, target_query, k=8, rot_weight=0.01):
        self.rot_weight = rot_weight
        
        if "pos" not in target_query.keys():
            target_query = get_pos(target_query, self.intrinsic)
        
        target_vec = self.get_vec_from_query(target_query)        
        dists, nears = self.neigh.kneighbors(target_vec.cpu(), n_neighbors=k, ) # B, k
        B, k = nears.shape
        
        near_queries = {}
        for key in self.all_query.keys():
            _, S, D = self.all_query[key].shape
            index_nears = repeat(torch.tensor(nears), "B k -> B k S D",S=S,D=D)
            index_nears = rearrange(index_nears, "B k S D -> (B k) S D") 
            
            retrieved_query = torch.gather(self.all_query[key], 0, index_nears)
            retrieved_query = rearrange(retrieved_query, "(B k) S D -> B k S D",B=B)
            near_queries[key] = retrieved_query
            
        return near_queries, nears, dists
    
    def weighted_euclidian_distance(self, query1, query2):
        pos1, pos2 = torch.from_numpy(query1[:3*self.seq_len]), torch.from_numpy(query2[:3*self.seq_len])
        rot1, rot2 = torch.from_numpy(query1[3*self.seq_len:]), torch.from_numpy(query2[3*self.seq_len:])

        pos1 = rearrange(pos1, "(S D) -> S D", S=self.seq_len)
        pos2 = rearrange(pos2, "(S D) -> S D", S=self.seq_len)
        rot1 = rearrange(rot1, "(S D) -> S D", S=self.seq_len)
        rot2 = rearrange(rot2, "(S D) -> S D", S=self.seq_len)

        pos_dis = torch.mean(torch.linalg.norm(pos1-pos2, dim=1))
        rot_dis = torch.mean(torch.linalg.norm(rot1-rot2, dim=1))
        return pos_dis + (self.rot_weight * rot_dis)

    def get_vec_from_query(self, query):
        """
        query: dict
        query["???"]: torch.array, shape:(Sequence_Length, Dim of ???), e.g, shape of query["pos"] = (101, 3)
        """
        pos = rearrange(query["pos"], "B S D -> B (S D)")
        rot = rearrange(query["rotation"], "B S D -> B (S D)")
        return torch.cat([pos, rot], 1)

class VAE_Retrieval():
    
    def __init__(self, dataset, VAE, n_neighbors=8):
        
        self.VAE = VAE
        self.VAE.eval()
        self.dataset_z = self.get_all_z(dataset)
        
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors)
        self.neigh.fit(self.dataset_z)
        
    def get_all_z(self, dataset):
        
        all_query = {}
        for i in range(len(dataset)):
            _, query = dataset[i]
            if i == 0:
                for key in query.keys():
                    all_query[key] = torch.unsqueeze(query[key], 0)
            else:
                for key in query.keys():
                    all_query[key] = torch.cat([all_query[key], torch.unsqueeze(query[key], 0)], 0)

        with torch.no_grad():
            all_z = self.VAE.encode(all_query)
            
        return all_z
    
    def retrieve_k_sample(self, target_query, k=8):
        
        with torch.no_grad():
            target_z = self.VAE.encode(target_query)
            
        dists, nears = self.neigh.kneighbors(target_z, n_neighbors=k) # B, k
        B, k = nears.shape
        nears_arranged = rearrange(nears, "B K -> (B K)")
        nears_z = self.dataset_z[nears_arranged]

        with torch.no_grad():
            near_queries = self.VAE.decode(nears_z)
            
        for key in near_queries.keys():
            near_queries[key] = rearrange(near_queries[key], "(B k) S D -> B k S D", B=B)
            
        return near_queries, nears, dists

class MSE_Based_Retrieval():
    
    def __init__(self, dataset, model):
            
        self.model = model
        self.model.eval()
        self.model.to("cuda")
        self.all_imgs, self.all_query = self.get_img_feature(dataset)
        self.MSE = torch.nn.MSELoss(reduction="none")
        
    def get_img_feature(self, dataset):
        
        all_query = {}
        all_imgs = []
        print("loading data")
        for i in tqdm(range(len(dataset))):
            image, query = dataset[i]
            image = torch.unsqueeze(image, 0).to("cuda")
            if i == 0:
                for key in query.keys():
                    all_query[key] = torch.unsqueeze(query[key], 0)
            else:
                for key in query.keys():
                    all_query[key] = torch.cat([all_query[key], torch.unsqueeze(query[key], 0)], 0)
            
            all_imgs.append(image)
        
        all_imgs = torch.cat(all_imgs, 0)
        
        return all_imgs, all_query
    
    def retrieve_k_sample(self, target_image, k=8, mini_batch=10):
        B, _, _, _ = target_image.shape

        with torch.no_grad():
            target_img_features = self.model.get_img_feature(target_image.to("cuda"))
            target_img_features = rearrange(target_img_features, "B C H W -> B (C H W)")
            target_img_features = repeat(target_img_features, "B D -> B N D", N=mini_batch)
        
        diff_list = []
        with torch.no_grad():
            for i in tqdm(range(0, 1000, mini_batch)):
                img_features = self.model.get_img_feature(self.all_imgs[i:i+mini_batch])
                img_features = rearrange(img_features, "N C H W -> N (C H W)")
                img_features = repeat(img_features, "N D -> B N D", B=B)
                diff = self.MSE(target_img_features, img_features)
                diff = torch.mean(diff, 2).to("cpu")
                diff_list.append(diff)
            
            diff = torch.cat(diff_list, 1)
            dists, nears = torch.sort(diff, 1)
            dists, nears = dists[:,:k], nears[:,:k]
        
        near_queries = {}
        for key in self.all_query.keys():
            _, S, D = self.all_query[key].shape
            index_nears = repeat(nears, "B k -> B k S D",S=S,D=D)
            index_nears = rearrange(index_nears, "B k S D -> (B k) S D") 
            
            retrieved_query = torch.gather(self.all_query[key], 0, index_nears)
            retrieved_query = rearrange(retrieved_query, "(B k) S D -> B k S D",B=B)
            near_queries[key] = retrieved_query
            
        return near_queries, nears, dists

class Image_Based_Retrieval_SPE():
    
    def __init__(self, dataset, model):
        
        self.model = model
        self.model.eval()
        self.model.to("cuda")
        self.img_features, self.all_query = self.get_img_feature(dataset)
        self.MSE = torch.nn.MSELoss(reduction="none")
        
    def get_img_feature(self, dataset):
        all_query = {}
        img_feature_list = []
        print("loading and preprocessing data")
        for i in tqdm(range(len(dataset))):
            img, query = dataset[i]

            img = torch.unsqueeze(img, 0)
            for key in query.keys():
                query[key] = torch.unsqueeze(query[key], 0)

            if i == 0:
                for key in query.keys():
                    all_query[key] = [query[key]]
            else:
                for key in query.keys():
                    all_query[key].append(query[key])

            with torch.no_grad():
                img = img.to("cuda")
                img_feature = self.model.get_extracted_img_feature(img, query).cpu()
                img_feature_list.append(img_feature)

        for key in all_query.keys():
            all_query[key] = torch.cat(all_query[key], 0)

        if img_feature_list[0].dim() == 3:
            img_features = torch.cat(img_feature_list, 1)
            img_features = rearrange(img_features, "S N D -> N (S D)")
        elif img_feature_list[0].dim() == 4:
            img_features = torch.cat(img_feature_list, 0)
            img_features = rearrange(img_features, "N C H W -> N (C H W)")

        return img_features, all_query
    
    def retrieve_k_sample(self, target_image, k=8):
        B, _, _, _ = target_image.shape

        target_img_feature_list = []
        for i in tqdm(range(len(self.img_features))):
            ins_query = {}

            for key in self.all_query.keys():
                ins_query[key] = repeat(self.all_query[key][i], "S D -> B S D", B=B)

            with torch.no_grad():
                if i == 0:
                    target_img_feature = self.model.get_extracted_img_feature(target_image.to("cuda"), ins_query).cpu() # S B D
                else:
                    target_img_feature = self.model.get_extracted_img_feature(target_image.to("cuda"), ins_query, with_feature=True).cpu() # S B D

            target_img_feature_list.append(target_img_feature)
        
        target_img_features = torch.stack(target_img_feature_list, 1) # S N B D

        if target_img_features.dim() == 4:
            target_img_features = rearrange(target_img_features, "S N B D -> B N (S D)")
        elif target_img_features.dim() == 5:
            target_img_features = rearrange(target_img_features, "B N C H W -> B N (C H W)")
        
        with torch.no_grad():
            diff = self.MSE(target_img_features, repeat(self.img_features, "N D -> B N D", B=B))
            diff = torch.mean(diff, 2)
            dists, nears = torch.sort(diff, 1)
            dists, nears = dists[:,:k], nears[:,:k]
        
        near_queries = {}
        for key in self.all_query.keys():
            _, S, D = self.all_query[key].shape
            index_nears = repeat(nears, "B k -> B k S D",S=S,D=D)
            index_nears = rearrange(index_nears, "B k S D -> (B k) S D") 
            
            retrieved_query = torch.gather(self.all_query[key], 0, index_nears)
            retrieved_query = rearrange(retrieved_query, "(B k) S D -> B k S D",B=B)
            near_queries[key] = retrieved_query
            
        _, D = self.img_features.shape
        index_nears = repeat(nears, "B k -> B k D",D=D)
        index_nears = rearrange(index_nears, "B k D -> (B k) D") 
        near_features = torch.gather(self.img_features, 0, index_nears)
        near_features = rearrange(near_features, "(B k) D -> B k D",B=B)
        return near_queries, nears, dists, near_features

class BYOL_Retrieval():
    
    def __init__(self, dataset, task_name, gn=True, wo_crop=False, pre_trained_dir="weights"):
        
        # get pre-trained model path
        model_name = f"BYOL_wo_crop_{wo_crop}"
        if gn:
            model_name = f"{model_name}_gn"
        pretrained_path = f"../{pre_trained_dir}/RLBench/{task_name}/{model_name}/model/model_iter10000.pth"
        print(f"BYOL_path: {pretrained_path}")
        
        # get model
        model = models.resnet50()
        if gn:
            model = self.replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, torch.nn.BatchNorm2d),
                func=lambda x: torch.nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
                )
        
        # load weights
        model, _, _, _, _ = load_checkpoint(model, pretrained_path)
        model.fc = torch.nn.Identity()
        self.model = model.eval()
        self.model.to("cuda")
        
        # set preprocess
        self.normalize = torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]))
        
        # get features
        self.img_features, self.all_query = self.get_img_feature(dataset)
        self.MSE = torch.nn.MSELoss(reduction="none")
        
    def get_img_feature(self, dataset):
        all_query = {}
        img_feature_list = []
        self.model.eval()
        print("loading and preprocessing data")
        for i in tqdm(range(len(dataset))):
            img, query = dataset[i]

            img = torch.unsqueeze(img, 0)
            for key in query.keys():
                query[key] = torch.unsqueeze(query[key], 0)

            if i == 0:
                for key in query.keys():
                    all_query[key] = [query[key]]
            else:
                for key in query.keys():
                    all_query[key].append(query[key])

            with torch.no_grad():
                img = img.to("cuda")
                img = img[:,:3]
                img_feature = self.model(self.normalize(img)).cpu()
                img_feature_list.append(img_feature)

        for key in all_query.keys():
            all_query[key] = torch.cat(all_query[key], 0)

        img_features = torch.cat(img_feature_list, 0)
        return img_features, all_query
    
    def retrieve_k_sample(self, target_image, k=8, rot_weight=0.01):
        B, _, _, _ = target_image.shape

        self.model.eval()
        with torch.no_grad():
            target_img_features = self.model(target_image[:,:3].to("cuda")).cpu() # B D
            target_img_features = self.model(self.normalize(target_image[:,:3].to("cuda"))).cpu() # B D
            target_img_features = repeat(target_img_features, "B D -> B N D", N=1000)
        
        with torch.no_grad():
            diff = self.MSE(target_img_features, repeat(self.img_features, "N D -> B N D", B=B))
            diff = torch.mean(diff, 2)
            dists, nears = torch.sort(diff, 1)
            dists, nears = dists[:,:k], nears[:,:k]
        
        near_queries = {}
        for key in self.all_query.keys():
            _, S, D = self.all_query[key].shape
            index_nears = repeat(nears, "B k -> B k S D",S=S,D=D)
            index_nears = rearrange(index_nears, "B k S D -> (B k) S D") 
            
            retrieved_query = torch.gather(self.all_query[key], 0, index_nears)
            retrieved_query = rearrange(retrieved_query, "(B k) S D -> B k S D",B=B)
            near_queries[key] = retrieved_query
            
        return near_queries, nears, dists
        
    @staticmethod
    def replace_submodules(
            root_module: torch.nn.Module, 
            predicate: Callable[[torch.nn.Module], bool], 
            func: Callable[[torch.nn.Module], torch.nn.Module]) -> torch.nn.Module:
        """
        from: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/vision/multi_image_obs_encoder.py
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
            if isinstance(parent_module, torch.nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, torch.nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all BN are replaced
        bn_list = [k.split('.') for k, m 
            in root_module.named_modules(remove_duplicate=True) 
            if predicate(m)]
        assert len(bn_list) == 0
        return root_module


class CLIP_Retrieval():
    
    def __init__(self, dataset, device="cuda"):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.model.to(device)
        self.topil = torchvision.transforms.ToPILImage()
        
        self.img_features, self.all_query = self.get_img_feature(dataset)
        self.MSE = torch.nn.MSELoss(reduction="none")
        
    def get_img_feature(self, dataset):
        all_query = {}
        img_feature_list = []
        print("loading and preprocessing data")
        for i in tqdm(range(len(dataset))):
            img, query = dataset[i]
            
            for key in query.keys():
                query[key] = torch.unsqueeze(query[key], 0)

            if i == 0:
                for key in query.keys():
                    all_query[key] = [query[key]]
            else:
                for key in query.keys():
                    all_query[key].append(query[key])

            with torch.no_grad():
                img = self.preprocess(self.topil(img[:3])).unsqueeze(0).to(self.device)
                img_feature = self.model.encode_image(img).cpu()
                img_feature_list.append(img_feature)

        for key in all_query.keys():
            all_query[key] = torch.cat(all_query[key], 0)

        img_features = torch.cat(img_feature_list, 0)
        return img_features, all_query
    
    def retrieve_k_sample(self, target_image, k=8):
        B, _, _, _ = target_image.shape
        target_image = torch.stack([self.preprocess(self.topil(target_image[i, :3])) for i in range(B)], 0)
        
        with torch.no_grad():
            target_img_features = self.model.encode_image(target_image.to("cuda")).cpu() # B D
            target_img_features = repeat(target_img_features, "B D -> B N D", N=1000)
            print(target_img_features.shape)
        
        with torch.no_grad():
            diff = self.MSE(target_img_features, repeat(self.img_features, "N D -> B N D", B=B))
            diff = torch.mean(diff, 2)
            dists, nears = torch.sort(diff, 1)
            dists, nears = dists[:,:k], nears[:,:k]
        
        near_queries = {}
        for key in self.all_query.keys():
            _, S, D = self.all_query[key].shape
            index_nears = repeat(nears, "B k -> B k S D",S=S,D=D)
            index_nears = rearrange(index_nears, "B k S D -> (B k) S D") 
            
            retrieved_query = torch.gather(self.all_query[key], 0, index_nears)
            retrieved_query = rearrange(retrieved_query, "(B k) S D -> B k S D",B=B)
            near_queries[key] = retrieved_query
            
        return near_queries, nears, dists