import torch
import math
from einops import rearrange, reduce, repeat

def evaluate_IBC(x, query, model, max_batch=50, device='cuda'):
    criteria = torch.nn.SmoothL1Loss()
    x = x.cpu()
    B, C, H, W = x.shape
    HW = H*W 
    image_size = W
    debug_info = {}
    with torch.inference_mode():
        size_range = torch.arange(image_size)

        x_range = repeat(size_range, 'h -> w h d', w=image_size, d=1)
        x_range = rearrange(x_range, 'w h d-> (w h) d')
        y_range = repeat(size_range, 'h -> h w d', w=image_size, d=1)
        y_range = rearrange(y_range, 'h w d-> (h w) d')

        pos = torch.cat([x_range, y_range], 1)
        pos = torch.unsqueeze(pos, 0)
        pos = pos.float()
        for b in range(B):
            start_index = 0
            while start_index < HW:
                if start_index + max_batch > HW:
                    end_index = HW
                else:
                    end_index = start_index + max_batch
                img = torch.unsqueeze(x[b], 0)
                
                num_sample = end_index - start_index
                color = repeat(query['color'][b], "C -> B N C",B=1, N=num_sample)
                dis = repeat(query['dis'][b], "C -> B N C",B=1, N=num_sample)
                input_dict = {"uv": pos[:, start_index:end_index].to(device), "color":color.to(device), "dis":dis.to(device)}
                
                if start_index == 0:
                    value_dict, pred_info = model(img.to(device), input_dict)
                    value = value_dict["all"]
                    value = value.cpu()
                    value_vec = value
                    
                    for key in pred_info.keys():
                        if key not in debug_info.keys():
                            debug_info[key] = pred_info[key]
                        elif key[:3] == "sep":
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 1)
                        elif key in debug_info.keys():
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 0)
                        else:
                            raise ValueError("error")
                else:
                    value_dict, pred_info = model(img.to(device), input_dict, with_feature=True)
                    value = value_dict["all"]
                    value = value.cpu()
                    value_vec = torch.cat((value_vec, value), 1)
                    for key in pred_info.keys():
                        if key[:3] == "sep":
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 1)
                    
                    
                start_index += max_batch
            
            if b == 0:
                b_value = torch.unsqueeze(value_vec, 0)
            else:
                b_value = torch.cat((b_value, torch.unsqueeze(value_vec, 0)), 0)
        
        min_value, min_index = b_value.min(dim=2)
        pos = pos[0]
        pred_pos = pos[min_index[:,0]]
        
        # change format from vec to image
        value_map = rearrange(b_value, 'b z (w h) -> b z w h', h=image_size)
        debug_info["value_map"] = value_map.detach().cpu()
        for key in debug_info.keys():
            if key[:3] == "sep":
                debug_info[key] = rearrange(debug_info[key], 'O (B w h) -> B O w h',h=image_size,w=image_size) # from 1, num_sample, batch -> batch, 1, num_sample
        
        # calculate loss
        loss = criteria(query['uv'].to(device), pred_pos.to(device))
        
        return pred_pos, debug_info, loss.detach().cpu()

def evaluate_IBC_multi(x, query, model, max_batch=50, device='cuda'):
    criteria = torch.nn.SmoothL1Loss()
    x = x.cpu()
    B, C, H, W = x.shape
    HW = H*W 
    image_size = W
    debug_info = {}
    
    # get one positive data
    for key in query.keys():
        query[key] = query[key][:,0]

    with torch.inference_mode():
        size_range = torch.arange(image_size)

        x_range = repeat(size_range, 'h -> w h d', w=image_size, d=1)
        x_range = rearrange(x_range, 'w h d-> (w h) d')
        y_range = repeat(size_range, 'h -> h w d', w=image_size, d=1)
        y_range = rearrange(y_range, 'h w d-> (h w) d')

        pos = torch.cat([x_range, y_range], 1)
        pos = torch.unsqueeze(pos, 0)
        pos = pos.float()
        for b in range(B):
            start_index = 0
            while start_index < HW:
                if start_index + max_batch > HW:
                    end_index = HW
                else:
                    end_index = start_index + max_batch
                img = torch.unsqueeze(x[b], 0)
                
                num_sample = end_index - start_index
                color = repeat(query['color'][b], "C -> B N C",B=1, N=num_sample)
                dis = repeat(query['dis'][b], "C -> B N C",B=1, N=num_sample)
                input_dict = {"uv": pos[:, start_index:end_index].to(device), "color":color.to(device), "dis":dis.to(device)}
                
                if start_index == 0:
                    value_dict, pred_info = model(img.to(device), input_dict)
                    value = value_dict["all"]
                    value = value.cpu()
                    value_vec = value
                    
                    for key in pred_info.keys():
                        if key not in debug_info.keys():
                            debug_info[key] = pred_info[key]
                        elif key[:3] == "sep":
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 1)
                        elif key in debug_info.keys():
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 0)
                        else:
                            raise ValueError("error")
                else:
                    value_dict, pred_info = model(img.to(device), input_dict, with_feature=True)
                    value = value_dict["all"]
                    value = value.cpu()
                    value_vec = torch.cat((value_vec, value), 1)
                    for key in pred_info.keys():
                        if key[:3] == "sep":
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 1)
                    
                    
                start_index += max_batch
            
            if b == 0:
                b_value = torch.unsqueeze(value_vec, 0)
            else:
                b_value = torch.cat((b_value, torch.unsqueeze(value_vec, 0)), 0)
        
        min_value, min_index = b_value.min(dim=2)
        pos = pos[0]
        pred_pos = pos[min_index[:,0]]
        
        # change format from vec to image
        value_map = rearrange(b_value, 'b z (w h) -> b z w h', h=image_size)
        debug_info["value_map"] = value_map.detach().cpu()
        for key in debug_info.keys():
            if key[:3] == "sep":
                debug_info[key] = rearrange(debug_info[key], 'O (B w h) -> B O w h',h=image_size,w=image_size) # from 1, num_sample, batch -> batch, 1, num_sample
        
        # calculate loss
        loss = criteria(query['uv'].to(device), pred_pos.to(device))
        
        return pred_pos, debug_info, loss.detach().cpu()

def evaluate_IBC_easy(x, query, model, max_batch=50, device='cuda'):
    criteria = torch.nn.SmoothL1Loss()
    x = x.cpu()
    B, C, H, W = x.shape
    HW = H*W 
    image_size = W
    debug_info = {}
    
    with torch.inference_mode():
        size_range = torch.arange(image_size)

        x_range = repeat(size_range, 'h -> w h d', w=image_size, d=1)
        x_range = rearrange(x_range, 'w h d-> (w h) d')
        y_range = repeat(size_range, 'h -> h w d', w=image_size, d=1)
        y_range = rearrange(y_range, 'h w d-> (h w) d')

        pos = torch.cat([x_range, y_range], 1)
        pos = torch.unsqueeze(pos, 0)
        pos = pos.float()
        for b in range(B):
            start_index = 0
            while start_index < HW:
                if start_index + max_batch > HW:
                    end_index = HW
                else:
                    end_index = start_index + max_batch
                img = torch.unsqueeze(x[b], 0)
                
                num_sample = end_index - start_index
                color = repeat(query['color'][b], "C -> B N C",B=1, N=num_sample)
                input_dict = {"uv": pos[:, start_index:end_index].to(device), "color":color.to(device)}
                
                if start_index == 0:
                    value_dict, pred_info = model(img.to(device), input_dict)
                    value = value_dict["all"]
                    value_vec = value
                    
                    for key in pred_info.keys():
                        if key not in debug_info.keys():
                            debug_info[key] = pred_info[key]
                        elif key[:3] == "sep":
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 1)
                        elif key in debug_info.keys():
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 0)
                        else:
                            raise ValueError("error")
                else:
                    value_dict, pred_info = model(img.to(device), input_dict, with_feature=True)
                    value = value_dict["all"]
                    value = value.cpu()
                    value_vec = torch.cat((value_vec, value), 1)
                    for key in pred_info.keys():
                        if key[:3] == "sep":
                            debug_info[key] = torch.cat([debug_info[key],pred_info[key]], 1)
                    
                    
                start_index += max_batch
            
            if b == 0:
                b_value = torch.unsqueeze(value_vec, 0)
            else:
                b_value = torch.cat((b_value, torch.unsqueeze(value_vec, 0)), 0)
        
        min_value, min_index = b_value.min(dim=2)
        pos = pos[0]
        pred_pos = pos[min_index[:,0]]
        
        # change format from vec to image
        value_map = rearrange(b_value, 'b z (w h) -> b z w h', h=image_size)
        debug_info["value_map"] = value_map.detach().cpu()
        for key in debug_info.keys():
            if key[:3] == "sep":
                debug_info[key] = rearrange(debug_info[key], 'O (B w h) -> B O w h',h=image_size,w=image_size) # from 1, num_sample, batch -> batch, 1, num_sample
        
        # calculate loss
        loss = criteria(query['uv'].to(device), pred_pos.to(device))
        
        return pred_pos, debug_info, loss.detach().cpu()

def evaluate_IBC_RLBench(cfg, x, query, obs, model, max_batch=50, device='cuda'):
    """
    We evaluate the model with following criteria
    1: Likelyhood of positive sample
    2: Change the one component of query, and get the difference between gt and predicted one
    """

    criteria = torch.nn.L1Loss()
    B, C, H, W = x.shape
    HW = H*W 
    image_size = W
    debug_info = {}
    debug_info["energy_map"] = {"time":[], "value":[]}
    debug_info["pred_pose"] = {"time":[], "value":[]}
    debug_info["gt_pose"] = {"time":[], "value":[]}
    eval_info_dict = {}

    with torch.inference_mode():
        ### evaluate positive sample ###
        energy, _ = model(x, query, obs)
        output = -1 * energy["all"]

        # get one positive data
        if cfg.PRED_LEN == 1:
            time_list = [0]
        else:
            raise ValueError("TODO change time at 338")
            time_list = [0, -1]
        
        for time_index in time_list:
            temp_query = {}
            for key in query.keys():
                temp_query[key] = query[key][:, time_index]

            ### get uv diff ###
            size_range = torch.arange(image_size)
            x_range = repeat(size_range, 'h -> w h d', w=image_size, d=1)
            x_range = rearrange(x_range, 'w h d-> (w h) d')
            x_range = (x_range / H * 2) - 1
            y_range = repeat(size_range, 'h -> h w d', w=image_size, d=1)
            y_range = rearrange(y_range, 'h w d-> (h w) d')
            y_range = (y_range / W * 2) - 1

            pos = torch.cat([x_range, y_range], 1)
            pos = torch.unsqueeze(pos, 0)
            pos = pos.float()

            b_value_list = []
            for b in range(B):
                start_index = 0
                value_list = []
                while start_index < HW:
                    if start_index + max_batch > HW:
                        end_index = HW
                    else:
                        end_index = start_index + max_batch
                    img = torch.unsqueeze(x[b], 0)
                    obs_b = torch.unsqueeze(obs[b], 0)
                    
                    num_sample = end_index - start_index
                    
                    input_dict = {}
                    for key in temp_query.keys():
                        if key == "uv":
                            input_dict[key] = pos[:, start_index:end_index].to(device)
                        else:
                            input_dict[key] = repeat(temp_query[key][b], "C -> B N C",B=1, N=num_sample).to(device)
                    
                    if start_index == 0:
                        value_dict, _ = model(img.to(device), input_dict, obs_b)
                    else:
                        value_dict, _ = model(img.to(device), input_dict, obs_b, with_feature=True)
                    
                    value_list.append(value_dict["all"].cpu())   
                    start_index += max_batch
                
                b_value_list.append(torch.cat(value_list, 1))
            b_value = torch.stack(b_value_list)
            min_value, min_index = b_value.min(dim=2)
            pos = pos[0]
            pred_pos = pos[min_index[:,0]]
            pred_pos = (pred_pos + 1) / 2 * torch.tensor([H, W])
            gt_pos = (temp_query['uv'] + 1) / 2 * torch.tensor([H, W]).to(device)
            
            # change format from vec to image
            energy_map = rearrange(b_value, 'b z (w h) -> b z w h', h=image_size)
            
            # calculate loss
            time = cfg.NEXT_LEN
            uv_diff = criteria(gt_pos.to(device), pred_pos.to(device))
            eval_info_dict["eval_uv_diff_t{}".format(time)] = uv_diff.detach().cpu()

            # register to dict
            debug_info["energy_map"]["time"].append(time)
            debug_info["energy_map"]["value"].append(energy_map.detach().cpu())

            debug_info["pred_pose"]["time"].append(time)
            debug_info["pred_pose"]["value"].append(pred_pos.detach().cpu())

            debug_info["gt_pose"]["time"].append(time)
            debug_info["gt_pose"]["value"].append(gt_pos.cpu())
        return debug_info, eval_info_dict

def random_sample(img_size, positive_data, num_sample):
    positive_uv = positive_data["uv"]
    positive_dis = positive_data["dis"]
    positive_color = positive_data["color"]
    B, _ = positive_uv.shape
    device = positive_uv.device

    label_dict = {}
    label_dict["color"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["color"][:,0] = 1.

    label_dict["dis"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["dis"][:,0] = 1.
    
    label_dict["uv"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["uv"][:,0] = 1.
    
    label_dict["all"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["all"][:,0] = 1.

    # get uv data
    negative_uv = torch.rand(B, num_sample, 2) * (img_size - 1)
    positive_uv = torch.unsqueeze(positive_uv, 1)
    uv = torch.cat((positive_uv, negative_uv.to(device)), 1)
    
    # get dis data
    max_distance = img_size * math.sqrt(2)
    negative_dis = torch.rand(B, num_sample, 1) * (max_distance - 1)
    positive_dis = torch.unsqueeze(positive_dis, 1)
    dis = torch.cat((positive_dis, negative_dis.to(device)), 1)

    # get color data
    color_choice = torch.tensor([[255.,0., 0.],[0.,0.,255.]])
    p = torch.tensor([0.5,0.5])
    index = p.multinomial(num_samples=B*num_sample, replacement=True)
    negative_color = color_choice[index]
    negative_color = rearrange(negative_color, '(B N) C -> B N C', B=B)
    positive_color = torch.unsqueeze(positive_color, 1)
    color = torch.cat((positive_color, negative_color.to(device)), 1)
    return {"uv": uv, "color": color, "dis": dis}, label_dict

def random_sample_multi(img_size, positive_data, num_sample):
    positive_uv = positive_data["uv"]
    positive_dis = positive_data["dis"]
    positive_color = positive_data["color"]
    B, N, _ = positive_uv.shape
    device = positive_uv.device

    label_dict = {}
    label_dict["color"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["color"][:,:N] = 1.

    label_dict["dis"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["dis"][:,:N] = 1.
    
    label_dict["uv"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["uv"][:,:N] = 1.
    
    label_dict["all"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["all"][:,:N] = 1.

    # get uv data
    positive_uv = positive_data["uv"]
    B, _, _ = positive_uv.shape
    device = positive_uv.device
    negative_uv = torch.rand(B, num_sample, 2) * (img_size - 1)
    # positive_uv = torch.unsqueeze(positive_uv, 1)
    uv = torch.cat((positive_uv, negative_uv.to(device)), 1)
    
    # get dis data
    positive_dis = positive_data["dis"]
    max_distance = img_size * math.sqrt(2)
    negative_dis = torch.rand(B, num_sample, 1) * (max_distance - 1)
    # positive_dis = torch.unsqueeze(positive_dis, 1)
    dis = torch.cat((positive_dis, negative_dis.to(device)), 1)

    # get color data
    positive_color = positive_data["color"]
    color_choice = torch.tensor([[255.,0., 0.],[0.,0.,255.]])
    p = torch.tensor([0.5,0.5])
    index = p.multinomial(num_samples=B*num_sample, replacement=True)
    negative_color = color_choice[index]
    negative_color = rearrange(negative_color, '(B N) C -> B N C', B=B)
    # positive_color = torch.unsqueeze(positive_color, 1)
    color = torch.cat((positive_color, negative_color.to(device)), 1)
    return {"uv": uv, "color": color, "dis": dis}, label_dict

def random_sample_easy(img_size, positive_data, num_sample):
    positive_uv = positive_data["uv"]
    positive_color = positive_data["color"]
    B, _ = positive_uv.shape
    device = positive_uv.device

    label_dict = {}
    label_dict["color"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["color"][:,0] = 1.

    label_dict["uv"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["uv"][:,0] = 1.
    
    label_dict["all"] = torch.zeros(B, 1+num_sample).to(device)
    label_dict["all"][:,0] = 1.

    # get uv data
    positive_uv = positive_data["uv"]
    B, _ = positive_uv.shape
    device = positive_uv.device
    negative_uv = torch.rand(B, num_sample, 2) * (img_size - 1)
    positive_uv = torch.unsqueeze(positive_uv, 1)
    uv = torch.cat((positive_uv, negative_uv.to(device)), 1)

    # get color data
    positive_color = positive_data["color"]
    color_choice = torch.tensor([[255.,0., 0.],[0.,0.,255.]])
    p = torch.tensor([0.5,0.5])
    index = p.multinomial(num_samples=B*num_sample, replacement=True)
    negative_color = color_choice[index]
    negative_color = rearrange(negative_color, '(B N) C -> B N C', B=B)
    positive_color = torch.unsqueeze(positive_color, 1)
    color = torch.cat((positive_color, negative_color.to(device)), 1)
    return {"uv": uv, "color": color}, label_dict

def grid_sample(img_size, positive_data, dis_grid_size=64, uv_grid_size=64, negative_color_size=8):
    """
    data
    index[0]: positive_data
    index[1:1+negative_color_size]: positive_uv, negative_color, positive_dis
    index[,1+negative_color_size+uv_grid_size]: positive_uv, positive_color, negative_dis
    index[,last]: negative_uv, random_color, random_dis
    """
    positive_color = positive_data["color"]
    positive_dis = positive_data["dis"]
    positive_uv = positive_data["uv"]
    device = positive_color.device
    B, _ = positive_color.shape

    label_dict = {}
    label_dict["color"] = torch.zeros(B, 1+negative_color_size+dis_grid_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["color"][:,0] = 1.
    label_dict["color"][:,1+negative_color_size:1+negative_color_size+dis_grid_size] = 1.

    label_dict["dis"] = torch.zeros(B, 1+negative_color_size+dis_grid_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["dis"][:,:1+negative_color_size] = 1.
    
    label_dict["uv"] = torch.zeros(B, 1+negative_color_size+dis_grid_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["uv"][:,:1+negative_color_size+dis_grid_size] = 1.
    
    label_dict["all"] = torch.zeros(B, 1+negative_color_size+dis_grid_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["all"][:,0] = 1.

    # positive_uv, negative_color, positive_dis
    #uv
    uv = repeat(positive_uv, "B P -> B N P", N=(negative_color_size+1))
    #color
    negative_color = positive_color[:,[2,1,0]]
    negative_color = repeat(negative_color, 'B C -> B N C', N=negative_color_size)
    color = torch.cat([torch.unsqueeze(positive_color, 1), negative_color.to(device)], 1)
    #dis
    dis = repeat(positive_dis, "B P -> B N P", N=(negative_color_size+1))

    # positive_uv, positive_color, negative_dis
    # uv
    uv = torch.cat([uv, repeat(positive_uv, "B P -> B N P", N=(dis_grid_size))], 1)
    # color
    color = torch.cat([color, repeat(positive_color, "B P -> B N P", N=(dis_grid_size))], 1)
    # dis
    max_distance = img_size * math.sqrt(2)
    grid_range = 1 / dis_grid_size
    grid_center = torch.arange(grid_range / 2, 1, grid_range)
    grid_center = repeat(grid_center, 'N -> B N P', B=B, P=1)
    grid_random = (torch.rand(B, dis_grid_size, 1) - 0.5) * grid_range
    grid_sample_dis = (grid_center + grid_random) * max_distance
    dis = torch.cat([dis, grid_sample_dis.to(device)], 1)

    # negative_uv, random_color, random_dis
    # uv
    grid_range = 1 / uv_grid_size
    grid_center_x = torch.arange(grid_range / 2, 1, grid_range)
    grid_center_x = repeat(grid_center_x, 'h -> w h d', w=uv_grid_size, d=1)
    grid_center_x = rearrange(grid_center_x, 'w h d-> (w h) d')
    grid_center_y = torch.arange(grid_range / 2, 1, grid_range)
    grid_center_y = repeat(grid_center_y, 'h -> h w d', w=uv_grid_size, d=1)
    grid_center_y = rearrange(grid_center_y, 'h w d -> (h w) d')
    grid_point = torch.cat([grid_center_x, grid_center_y], 1)
    grid_point = repeat(grid_point, 'N P -> B N P', B=B)
    grid_random = (torch.rand(B, uv_grid_size*uv_grid_size, 2) - 0.5) * grid_range
    grid_point = (grid_point + grid_random) * img_size
    uv = torch.cat((uv, grid_point.to(device)), 1)
    # color
    color_choice = torch.tensor([[255.,0., 0.],[0.,0.,255.]])
    p = torch.tensor([0.5,0.5])
    index = p.multinomial(num_samples=B*uv_grid_size*uv_grid_size, replacement=True)
    negative_color = color_choice[index]
    negative_color = rearrange(negative_color, '(B N) C -> B N C', B=B)
    color = torch.cat((color, negative_color.to(device)), 1)
    # dis
    max_distance = img_size * math.sqrt(2)
    negative_dis = torch.rand(B, uv_grid_size*uv_grid_size, 1) * (max_distance - 1)
    dis = torch.cat([dis, negative_dis.to(device)], 1)

    return {"uv": uv, "color":color, "dis":dis}, label_dict

def grid_sample_multi(img_size, positive_data, dis_grid_size=64, uv_grid_size=64, negative_color_size=8):
    """
    data
    index[:N]: positive_data
    index[N:N+negative_color_size*N]: positive_uv, negative_color, positive_dis
    index[,N+negative_color_size*N+uv_grid_size*N]: positive_uv, positive_color, negative_dis
    index[,last]: negative_uv, random_color, random_dis
    """
    positive_color = positive_data["color"]
    positive_dis = positive_data["dis"]
    positive_uv = positive_data["uv"]
    device = positive_color.device
    B, N, _ = positive_color.shape

    I0 = N
    I1 = I0 + (negative_color_size * N)
    I2 = I1 + (dis_grid_size * N)
    I3 = I2 + (uv_grid_size*uv_grid_size)

    label_dict = {}
    label_dict["color"] = torch.zeros(B, I3).to(device)
    label_dict["color"][:,:I0] = 1.
    label_dict["color"][:,I1:I2] = 1.

    label_dict["dis"] = torch.zeros(B, I3).to(device)
    label_dict["dis"][:,:I1] = 1.
    
    label_dict["uv"] = torch.zeros(B, I3).to(device)
    label_dict["uv"][:,:I2] = 1.
    
    label_dict["all"] = torch.zeros(B, I3).to(device)
    label_dict["all"][:,:I0] = 1.

    # positive_uv, negative_color, positive_dis
    #uv
    negative_uv = repeat(positive_uv, "B N P -> B (N D) P", D=negative_color_size)
    uv = torch.cat([positive_uv, negative_uv], 1)
    #color
    negative_color = positive_color[:,:,[2,1,0]]
    negative_color = repeat(negative_color, 'B N C -> B (N D) C', D=negative_color_size)
    color = torch.cat([positive_color, negative_color.to(device)], 1)
    #dis
    negative_dis = repeat(positive_dis, "B N P -> B (N D) P", D=negative_color_size)
    dis = torch.cat([positive_dis, negative_dis], 1)

    # positive_uv, positive_color, negative_dis
    # uv
    uv = torch.cat([uv, repeat(positive_uv, "B N P -> B (N D) P", D=(dis_grid_size))], 1)
    # color
    color = torch.cat([color, repeat(positive_color, "B N P -> B (N D) P", D=(dis_grid_size))], 1)
    # dis
    max_distance = img_size * math.sqrt(2)
    grid_range = 1 / dis_grid_size
    grid_center = torch.arange(grid_range / 2, 1, grid_range)
    grid_center = repeat(grid_center, 'D -> B (N D) P', B=B, N=N, P=1)
    grid_random = (torch.rand(B, N*dis_grid_size, 1) - 0.5) * grid_range
    grid_sample_dis = (grid_center + grid_random) * max_distance
    dis = torch.cat([dis, grid_sample_dis.to(device)], 1)

    # negative_uv, random_color, random_dis
    # uv
    grid_range = 1 / uv_grid_size
    grid_center_x = torch.arange(grid_range / 2, 1, grid_range)
    grid_center_x = repeat(grid_center_x, 'h -> w h d', w=uv_grid_size, d=1)
    grid_center_x = rearrange(grid_center_x, 'w h d-> (w h) d')
    grid_center_y = torch.arange(grid_range / 2, 1, grid_range)
    grid_center_y = repeat(grid_center_y, 'h -> h w d', w=uv_grid_size, d=1)
    grid_center_y = rearrange(grid_center_y, 'h w d -> (h w) d')
    grid_point = torch.cat([grid_center_x, grid_center_y], 1)
    grid_point = repeat(grid_point, 'N P -> B N P', B=B)
    grid_random = (torch.rand(B, uv_grid_size*uv_grid_size, 2) - 0.5) * grid_range
    grid_point = (grid_point + grid_random) * img_size
    uv = torch.cat((uv, grid_point.to(device)), 1)
    # color
    color_choice = torch.tensor([[255.,0., 0.],[0.,0.,255.]])
    p = torch.tensor([0.5,0.5])
    index = p.multinomial(num_samples=B*uv_grid_size*uv_grid_size, replacement=True)
    negative_color = color_choice[index]
    negative_color = rearrange(negative_color, '(B N) C -> B N C', B=B)
    color = torch.cat((color, negative_color.to(device)), 1)
    # dis
    max_distance = img_size * math.sqrt(2)
    negative_dis = torch.rand(B, uv_grid_size*uv_grid_size, 1) * (max_distance - 1)
    dis = torch.cat([dis, negative_dis.to(device)], 1)

    return {"uv": uv, "color":color, "dis":dis}, label_dict

def grid_sample_easy(img_size, positive_data, uv_grid_size=64, negative_color_size=64):
    """
    data
    index[0]: positive_data
    index[1:1+negative_color_size]: positive_uv, negative_color, positive_dis
    index[,1+negative_color_size+uv_grid_size]: positive_uv, positive_color, negative_dis
    index[,last]: negative_uv, random_color, random_dis
    """
    positive_color = positive_data["color"]
    positive_uv = positive_data["uv"]
    device = positive_color.device
    B, _ = positive_color.shape

    label_dict = {}
    label_dict["color"] = torch.zeros(B, 1+negative_color_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["color"][:,0] = 1.

    label_dict["uv"] = torch.zeros(B, 1+negative_color_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["uv"][:,:1+negative_color_size] = 1.
    
    label_dict["all"] = torch.zeros(B, 1+negative_color_size+(uv_grid_size*uv_grid_size)).to(device)
    label_dict["all"][:,0] = 1.

    # positive_uv, negative_color
    #uv
    uv = repeat(positive_uv, "B P -> B N P", N=(negative_color_size+1))
    #color
    negative_color = positive_color[:,[2,1,0]]
    negative_color = repeat(negative_color, 'B C -> B N C', N=negative_color_size)
    color = torch.cat([torch.unsqueeze(positive_color, 1), negative_color.to(device)], 1)

    # negative_uv, random_color
    # uv
    grid_range = 1 / uv_grid_size
    grid_center_x = torch.arange(grid_range / 2, 1, grid_range)
    grid_center_x = repeat(grid_center_x, 'h -> w h d', w=uv_grid_size, d=1)
    grid_center_x = rearrange(grid_center_x, 'w h d-> (w h) d')
    grid_center_y = torch.arange(grid_range / 2, 1, grid_range)
    grid_center_y = repeat(grid_center_y, 'h -> h w d', w=uv_grid_size, d=1)
    grid_center_y = rearrange(grid_center_y, 'h w d -> (h w) d')
    grid_point = torch.cat([grid_center_x, grid_center_y], 1)
    grid_point = repeat(grid_point, 'N P -> B N P', B=B)
    grid_random = (torch.rand(B, uv_grid_size*uv_grid_size, 2) - 0.5) * grid_range
    grid_point = (grid_point + grid_random) * img_size
    uv = torch.cat((uv, grid_point.to(device)), 1)
    # color
    color_choice = torch.tensor([[255.,0., 0.],[0.,0.,255.]])
    p = torch.tensor([0.5,0.5])
    index = p.multinomial(num_samples=B*uv_grid_size*uv_grid_size, replacement=True)
    negative_color = color_choice[index]
    negative_color = rearrange(negative_color, '(B N) C -> B N C', B=B)
    color = torch.cat((color, negative_color.to(device)), 1)

    return {"uv": uv, "color":color}, label_dict