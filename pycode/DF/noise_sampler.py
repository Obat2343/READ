import torch
from einops import rearrange, repeat

def Normal_Noise():

    def sampler(query):
        """
        Args
            query (dict): motion query or latent query. This dict should include tensors.
        """
        noise_dict = {}
        for key in query.keys():
            noise_dict[key] = torch.randn_like(query[key])
            noise_dict[key] = noise_dict[key].to(query[key].device)
        return noise_dict
    
    return sampler

class Latent_Noise_from_kNN():

    def __init__(self, training_dataset, vae, rank, device="cuda"):
        
        self.vae = vae
        self.vae.eval()
        self.device = device
        self.rank = rank
        
        self.whole_queries = self.get_all_queries(training_dataset)
        with torch.no_grad():
            self.whole_latents = vae.encode(self.whole_queries)
    
    def get_all_queries(self, dataset):
        dataset.without_img = True
        whole_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=16)
        for data in whole_dataloader:
            _, whole_queries = data

        for key in whole_queries.keys():
            whole_queries[key] = whole_queries[key].to(self.device)

        dataset.without_img = False
        return whole_queries

    def __call__(self, query, remove_first=True, rank=0):
        """
        Args
            query (dict): motion query or latent query. This dict should include tensors.
            remove_first (Bool): If true, remove 1st nearest neighbor sample because it is often same as the input query. 
            
        """
        # embed to latent
        if "latent" not in query.keys():
            with torch.no_grad():
                query = self.vae.encode(query)

        # calculate distances
        distances = torch.cdist(query["latent"], self.whole_latents["latent"])

        # sorting
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        # get random index
        if remove_first:
            base = 1
        else:
            base = 0

        if rank == 0:
            rank = self.rank

        B = query["latent"].shape[0]
        random_indices = torch.randint(base, base+rank, (B,1), device=self.device)

        # retrieve data
        retrieval_indices = torch.gather(sorted_indices, 1, random_indices).squeeze(1)
        retrieved_latents = self.whole_latents["latent"][retrieval_indices]

        noise = {}
        noise["latent"] = retrieved_latents - query["latent"]
        return noise
    
class Latent_Noise_from_All_kNN():

    def __init__(self, training_dataset, vae, rank, device="cuda"):
        
        self.vae = vae
        self.vae.eval()
        self.device = device
        self.rank = rank
        
        self.whole_queries = self.get_all_queries(training_dataset)
        with torch.no_grad():
            self.whole_latents = vae.encode(self.whole_queries)

        print("prepare noise database")
        self.noise_vectors = self.prepare_noise()
        print("finish")

    def get_all_queries(self, dataset):
        dataset.without_img = True
        whole_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=16)
        for data in whole_dataloader:
            _, whole_queries = data

        for key in whole_queries.keys():
            whole_queries[key] = whole_queries[key].to(self.device)

        dataset.without_img = False
        return whole_queries

    def prepare_noise(self):
        # 
        distances = torch.cdist(self.whole_latents["latent"], self.whole_latents["latent"])

        # 
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        # tensor2の形状(A, C)
        flat_indices = rearrange(sorted_indices[:,1:self.rank+1], "B N -> (B N)")
        retrieved_latents = self.whole_latents["latent"][flat_indices]
        retrieved_latents = rearrange(retrieved_latents, "(B N) D -> B N D", B=1000)

        original_latents = repeat(self.whole_latents["latent"], "B D -> B N D",N=self.rank)

        noise_vecs = rearrange(retrieved_latents - original_latents, "B N D -> (B N) D")
        return noise_vecs


    def __call__(self, query=None, remove_first=False):
        """
        Args
            query (dict): motion query or latent query. This dict should include tensors.
            remove_first (Bool): Dummy.
            
        """
        temp_key = list(query.keys())[0]
        B = query[temp_key].shape[0]

        noise_indices = torch.randint(0, len(self.noise_vectors), (B,))
        
        noise = {}
        noise["latent"] = self.noise_vectors[noise_indices].to(self.device)

        return noise

class Latent_Noise_from_AugkNN():

    def __init__(self, training_dataset, vae, rank=3, n_gather_vec=3, graph_size=3, device="cuda"):
        
        self.vae = vae
        self.vae.eval()
        self.device = device
        self.rank = rank
        self.n_gather_vec = n_gather_vec
        self.graph_size = graph_size
        
        self.whole_queries = self.get_all_queries(training_dataset)
        with torch.no_grad():
            self.whole_latents = self.vae.encode(self.whole_queries)
        
        self.noise_vectors_graph = self.prepare_noise()
    
    def get_all_queries(self, dataset):
        dataset.without_img = True
        whole_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=16)
        for data in whole_dataloader:
            _, whole_queries = data

        for key in whole_queries.keys():
            whole_queries[key] = whole_queries[key].to(self.device)

        dataset.without_img = False
        return whole_queries

    @torch.no_grad()
    def prepare_noise(self):
        # compute kNN 
        distances = torch.cdist(self.whole_latents["latent"], self.whole_latents["latent"])
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        # retrieve kNN points
        flat_indices = rearrange(sorted_indices[:,1:self.rank+1], "B N -> (B N)")
        retrieved_latents = self.whole_latents["latent"][flat_indices]
        retrieved_latents = rearrange(retrieved_latents, "(B N) D -> B N D", B=1000)

        # compute vectors to kNN points from original points
        original_latents = repeat(self.whole_latents["latent"], "B D -> B N D",N=self.rank)
        noise_vecs = retrieved_latents - original_latents # B N D

        # gathering vectors
        noise_vecs = torch.cat([noise_vecs[sorted_indices[:,i]] for i in range(0, self.n_gather_vec+1)], 1) # B  S*N D

        # create graph
        distances = torch.cdist(noise_vecs, noise_vecs) # B (S*N) (S*N)
        sorted_distances, sorted_indices = torch.sort(distances, dim=2) # B (S*N) (S*N)
        
        B, N, D = noise_vecs.shape
        reshaped_sorted_indices = rearrange(sorted_indices, "B N M -> B (N M)")
        reshaped_sorted_indices = repeat(reshaped_sorted_indices, "B Z -> B Z D", D=D)

        noise_vecs_graph = torch.gather(noise_vecs, 1, reshaped_sorted_indices) # B (S*N*S*N) D
        noise_vecs_graph = rearrange(noise_vecs_graph, "B (N M) D -> B N M D", N=N)
        return noise_vecs_graph[:,:,:self.graph_size] # B (n_gather_vecs) (graph size) D
    
    @torch.no_grad()
    def __call__(self, query, remove_first=False, rank=0):
        """
        Args
            query (dict): motion query or latent query. This dict should include tensors.
            remove_first (Bool): Dummy.
        """
        # embed to latent
        if "latent" not in query.keys():
            query = self.vae.encode(query)

        # calculate distances
        distances = torch.cdist(query["latent"], self.whole_latents["latent"])

        # sorting
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        
        retrieved_noise_vecs_graph = self.noise_vectors_graph[sorted_indices[:,0]] # B N G D
        B, N, G, D = retrieved_noise_vecs_graph.shape

        noise_indices = torch.randint(0, N, (B,1), device=self.device)
        noise_indices = repeat(noise_indices, "B N -> B N G D", G=G, D=D)
        
        selected_graph = torch.gather(retrieved_noise_vecs_graph, 1, noise_indices)[:,0].to(self.device) # B G D

        # sampling points inside the graph
        random_weight = torch.rand(B, G, device=self.device)
        total = torch.sum(random_weight, dim=1, keepdim=True)
        normalized_random_weight = repeat(random_weight / total, "B G -> B G D", D=D)
        vecs = torch.sum(normalized_random_weight * selected_graph, dim=1) # B D

        noise = {}
        noise["latent"] = vecs
        return noise