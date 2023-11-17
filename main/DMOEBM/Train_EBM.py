import argparse
import sys
import os
import yaml
import time
import shutil
import copy
import datetime

import torch
import wandb

sys.path.append("../")
from pycode.config import _C as cfg
from pycode.dataset import RLBench_DMOEBM
from pycode.misc import str2bool, save_checkpoint, visualize_inf_query, Timer, Time_memo, convert_rotation_6d_to_matrix, save_args, cat_pos_and_neg, visualize_negative_sample, load_checkpoint
from pycode.DMOEBM.total_model import DMOEBM
from pycode.DMOEBM.sampling import inference, Sampler, get_statistics_info
from pycode.DMOEBM.EBM_loss import EBM_Loss, Eval_score
from pycode.DMOEBM.Regression_loss import Evaluation
from pycode.DMOEBM.Motion_Gen import VAE, Single_Class_TransformerVAE

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/Transformer_EBM.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--save_data', type=str2bool, default=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--reset_dataset', type=str2bool, default=False)
parser.add_argument('--frame', type=int, default=100)
parser.add_argument('--rot_mode', type=str, default="6d")
parser.add_argument('--verbose',type=str2bool, default=False)
parser.add_argument('--pretrain',type=str2bool, default=False)
parser.add_argument('--tasks', nargs="*", type=str, default=["none"]) # PutRubbishInBin StackWine CloseBox PushButton ReachTarget TakePlateOffColoredDishRack PutKnifeOnChoppingBoard StackBlocks
args = parser.parse_args()

##### config #####
# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

    # set config_file to wandb
    with open(args.config_file) as file:
        obj = yaml.safe_load(file)

if args.tasks[0] != "none":
    task_list = args.tasks
else:
    task_list = [cfg.DATASET.RLBENCH.TASK_NAME]

# keep configs during training
base_yamlname = os.path.basename(args.config_file)
head, ext = os.path.splitext(args.config_file)
dt_now = datetime.datetime.now()
temp_yaml_path = f"{head}_{dt_now.year}{dt_now.month}{dt_now.day}_{dt_now.hour}:{dt_now.minute}:{dt_now.second}{ext}"
if args.log2wandb:
    shutil.copy(os.path.abspath(args.config_file), temp_yaml_path)

for task_name in task_list:
    cfg.DATASET.RLBENCH.TASK_NAME = task_name

    if ("vae" in cfg.SAMPLING.FIRST_SAMPLE) and (cfg.VAE.NAME == "Transformer_VAE"):
        sample_name = f"Transformer_vae_{cfg.VAE.LATENT_DIM}" + cfg.SAMPLING.FIRST_SAMPLE[3:]
    elif ("vae" in cfg.SAMPLING.FIRST_SAMPLE) and (cfg.VAE.NAME == "VAE"):
        sample_name = f"vae_{cfg.VAE.LATENT_DIM}" + cfg.SAMPLING.FIRST_SAMPLE[3:]
    else:
        sample_name = cfg.SAMPLING.FIRST_SAMPLE

    if args.name == "":
        dir_name = f"EBM_aug_frame_{args.frame}_mode_{args.rot_mode}_first_{sample_name}_second_{cfg.SAMPLING.SECOND_SAMPLE}_inf_{cfg.SAMPLING.INF_SAMPLE}"
    else:
        dir_name = args.name

    if args.add_name != "":
        dir_name = f"{dir_name}_{args.add_name}"

    if args.pretrain:
        dir_name = dir_name + "_pretrain"

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
        run = wandb.init(project='IBC-{}-{}'.format(cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_NAME), entity='tendon',
                        config=obj, save_code=True, name=dir_name, dir=save_dir)

    model_save_dir = os.path.join(save_path, "model")
    log_dir = os.path.join(save_path, 'log')
    vis_dir = os.path.join(save_path, 'vis')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # copy source code
    shutil.copy(sys.argv[0], save_path)
    if (len(args.config_file) > 0) and args.log2wandb:
        shutil.copy(temp_yaml_path, os.path.join(save_path, base_yamlname))

    # save args
    argsfile_path = os.path.join(save_path, "args.json")
    save_args(args,argsfile_path)

    # set dataset
    train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=8)
    for data in temp_loader:
        _, inf_query = data

    val_dataset = RLBench_DMOEBM("val", cfg, save_dataset=args.reset_dataset, num_frame=args.frame, rot_mode=rot_mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)

    # set model
    conv_dims = cfg.MODEL.CONV_DIMS
    enc_depths = cfg.MODEL.ENC_DEPTHS
    enc_layers = cfg.MODEL.ENC_LAYERS

    dec_depths = cfg.MODEL.DEC_DEPTHS
    dec_layers = cfg.MODEL.DEC_LAYERS

    extractor_name = cfg.MODEL.EXTRACTOR_NAME
    predictor_name = cfg.MODEL.PREDICTOR_NAME

    conv_droppath_rate = cfg.MODEL.CONV_DROP_PATH_RATE
    atten_dropout_rate = cfg.MODEL.ATTEN_DROPOUT_RATE
    query_emb_dim = cfg.MODEL.QUERY_EMB_DIM
    num_atten_block = cfg.MODEL.NUM_ATTEN_BLOCK

    model = DMOEBM(["uv","z","rotation","grasp_state","time"], [2,1,rot_dim,1,1], 
                        dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                        extractor_name=extractor_name, predictor_name=predictor_name, num_attn_block=num_atten_block, query_emb_dim=query_emb_dim,
                        drop_path_rate=conv_droppath_rate, mlp_drop=atten_dropout_rate)
    model = model.to(device)

    if args.pretrain:
        pretrain_name = f"EBM_aug_frame_{args.frame}_mode_{args.rot_mode}_first_random_pick_second_none_inf_{cfg.SAMPLING.INF_SAMPLE}"
        pretrain_path = f"../global_result/RLBench/{cfg.DATASET.RLBENCH.TASK_NAME}/{pretrain_name}/model/model_iter100000.pth"
        print(f"pretrain path: {pretrain_name}")
        if not os.path.exists(pretrain_path):
            raise ValueError("model is not exists")

        model, _, _, _, _ = load_checkpoint(model, pretrain_path)

    # set optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)
    Train_loss = EBM_Loss(device, mode=rot_mode)
    Eval_score_loss = Eval_score("cuda")
    Eval_error_loss = Evaluation('cuda',mode=rot_mode)

    # set other configuration
    info_dict = get_statistics_info(train_dataset)
    first_sample = cfg.SAMPLING.FIRST_SAMPLE
    second_sample = cfg.SAMPLING.SECOND_SAMPLE
    inf_sample = cfg.SAMPLING.INF_SAMPLE
    num_negative = cfg.SAMPLING.NUM_NEGATIVE

    if "langevin_vae" in second_sample:
        vae_pretrained_path = f"../global_result/RLBench/{cfg.DATASET.RLBENCH.TASK_NAME}/{cfg.VAE.NAME}_frame_{args.frame}_latentdim_{cfg.VAE.LATENT_DIM}_mode_{args.rot_mode}/model/model_iter100000.pth"
        if not os.path.exists(vae_pretrained_path):
            raise ValueError("please train vae first")

        input_size = (2 + 1 + rot_dim + 1) * (args.frame + 1)
        if cfg.VAE.NAME == "VAE":
            vae = VAE(input_size, cfg.VAE.LATENT_DIM, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
        elif cfg.VAE.NAME == "Transformer_VAE":
            vae = Single_Class_TransformerVAE(["uv","z","rotation","grasp_state"],[2,1,rot_dim,1], args.frame + 1, latent_dim=cfg.VAE.LATENT_DIM, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
        
        vae, _, _, _, _ = load_checkpoint(vae, vae_pretrained_path)
        vae.eval()

    else:
        vae = "none"

    negative_query_sampler = Sampler(first_sample, second_sample, cfg, info_dict, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode, frame=args.frame, vae=vae)

    iteration = 0
    start = time.time()
    time_memo = Time_memo()
    flag = False
    for epoch in range(10000000):
        for data in train_dataloader:
            image, h_query = data
            
            for key in h_query.keys():
                h_query[key] = h_query[key].to(device)
            image = image.to(device)

            if iteration < cfg.SAMPLING.SECOND_SAMPLE_STEP:
                negative_query = negative_query_sampler(num_negative, image, ebm_model=model, vae_model=vae, pos_sample=h_query, verbose=args.verbose, valid_second_sample=False)
            else:
                negative_query = negative_query_sampler(num_negative, image, ebm_model=model, vae_model=vae, pos_sample=h_query, verbose=args.verbose)

            query = cat_query = cat_pos_and_neg(h_query, negative_query)

            optimizer.zero_grad()

            with Timer() as t:
                pred_dict, info = model(image, query)
            time_memo.add("forward", t.secs)

            loss, loss_dict = Train_loss(pred_dict)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if iteration % log_iter == 0:
                end = time.time()
                cost = (end - start) / (iteration+1)
                print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss_dict["train/loss"]:.4g}')
                
                if args.log2wandb:
                    wandb.log(loss_dict, step=iteration)
                    wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=iteration)

            if iteration % eval_iter == 0:
                # save negative sample exaple
                image = image.cpu()
                for key in negative_query.keys():
                    negative_query[key] = negative_query[key][:8,:16]
                vis_img = visualize_negative_sample(image, negative_query, train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode)

                if args.save_data:
                    vis_img.save(os.path.join(vis_dir, f"trainning_negative_sample_{iteration}.png"))

                with torch.no_grad():

                    for data in val_dataloader:
                        model.eval()
                        image, gt_query = data
                        for key in gt_query.keys():
                            gt_query[key] = torch.unsqueeze(gt_query[key][:10].to(device), 1)
                        image = image[:10].to(device)

                        # get sample and score
                        sample, query_pred_dict = inference(inf_sample, copy.deepcopy(inf_query), cfg, image, model, info_dict=info_dict, intrinsic=train_dataset.info_dict["data_list"][0]["camera_intrinsic"], device=device)

                        # get gt score
                        gt_pred_dict, info = model(image, gt_query)

                        # evaluate score
                        score_loss_dict = Eval_score_loss(query_pred_dict, gt_pred_dict, mode="val")

                        # calculate pose error
                        top_query = {}
                        for key in sample.keys():
                            top_query[key] = sample[key][:,0]
                            gt_query[key] = torch.squeeze(gt_query[key], 1)

                        if rot_mode == "6d":
                            gt_query_6d, top_query = convert_rotation_6d_to_matrix([copy.deepcopy(gt_query), top_query])

                        error_loss_dict = Eval_error_loss(top_query, gt_query_6d, mode="val")
                        break

                    print(f'Val: {iteration} Pos:{error_loss_dict["val/pos_error"]:.4g}, z:{error_loss_dict["val/z_error"]:.4g}, rot:{error_loss_dict["val/rot_error"]:.4g}, grasp:{error_loss_dict["val/grasp_accuracy"]:.4g}')
                    optimizer.zero_grad()

                    # visualize
                    for key in gt_query.keys():
                        gt_query[key] = torch.squeeze(gt_query[key].cpu(), 1)
                    image = image.cpu()
                    vis_img = visualize_inf_query(5, 16, sample, gt_query, image, train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode, pred_score=query_pred_dict["score"], gt_score=gt_pred_dict["score"])
                    if args.save_data:
                        vis_img.save(os.path.join(vis_dir, f"pos_img_val_{iteration}.png"))

                    if args.log2wandb:
                        wandb.log(score_loss_dict, step=iteration)
                        wandb.log(error_loss_dict, step=iteration)

                model.train()

            if (iteration % save_iter == 0) and args.save_data:
                # save model
                model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
                save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

            if iteration == max_iter + 1:
                flag = True
                wandb.finish()
                break
            
            iteration += 1
        
        if flag == True:
            break

os.remove(temp_yaml_path)