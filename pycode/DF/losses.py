# This code is based on the original sde paper and modified to change the input from an image to a motion.
# Original paper and code are following:
#
# Paper:
# @inproceedings{
#     song2021scorebased,
#     title={Score-Based Generative Modeling through Stochastic Differential Equations},
#     author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
#     booktitle={International Conference on Learning Representations},
#     year={2021},
#     url={https://openreview.net/forum?id=PxTIG12RRHS}
# }
#
# Code:
# https://github.com/yang-song/score_sde_pytorch/tree/main

"""All functions related to loss computation and optimization.
"""

import warnings
import torch
import torch.optim as optim
import numpy as np
from einops import rearrange
from ..DF import utils
from ..DF.sde_lib import VESDE, VPSDE, IRSDE

def get_loss_fn(sde, target="auto", train=True, noise_sampler=None, energy=False, reduce_mean=False, continuous=True, likelihood_weighting=False, model_type="m"):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        tareget: String. If "auto", target is automatically select among ["noise", "gt_motion"].
        train: Bool.
        noise_sampler: A function to sample noise
        energy: please ignore this. If True, the diffusion model output the scaler instead vector. Then, vector is obtained by computing the gradient of scaler value respect to motion.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
        mdoel_type: output type

    Returns:
        A one-step function for training or evaluation.
    """

    if target == "auto":
        if model_type in ["m", "l"]:
            target = "noise"
        else:
            target = "gt_motion"

    if continuous:
        assert not energy, "not implemented (TODO)"

        if model_type == "m":
            if noise_sampler != None:
                raise NotImplementedError()
            
            if target == "noise":
                print("LOSS: sde loss")
                loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
            elif target == "gt_motion":
                print("LOSS: Cold Diffusion loss")
                if likelihood_weighting==True:
                    warnings.warn("Likelihood weighting is not supported for ColdDiffusion_like_loss", UserWarning)
                loss_fn = get_motion_loss_fn(sde, train, reduce_mean=reduce_mean)
            
        elif model_type == "l":
            if noise_sampler == None:
                raise NotImplementedError()
            
            if target == "noise":
                print("LOSS: latent sde loss")
                loss_fn = get_latent_sde_loss_fn(sde, train, noise_sampler, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
            elif target == "gt_motion":
                print("LOSS: Latent Cold Diffusion loss")
                if likelihood_weighting==True:
                    warnings.warn("Likelihood weighting is not supported for ColdDiffusion_like_loss", UserWarning)
                loss_fn = get_latent_motion_loss_fn(sde, train, noise_sampler, reduce_mean=reduce_mean)

        elif model_type == "l-m":
            if likelihood_weighting==True:
                warnings.warn("Likelihood weighting is not supported for ColdDiffusion_like_loss", UserWarning)
            
            if target == "noise":
                raise ValueError("target should be gt_motion for l-m type model")
            elif target == "gt_motion":
                print("LOSS: latent Cold Diffusion loss for l-m")
                loss_fn = get_latent_motion_loss_fn_for_lm(sde, train, noise_sampler, reduce_mean=reduce_mean)
            
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE) and energy:
            raise NotImplementedError("TODO")
        elif isinstance(sde, VESDE) and not energy:
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE) and energy:
            loss_fn = get_energy_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE) and not energy:
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    return loss_fn

def get_loss_fn_CG(sde, target="auto", train=True, noise_sampler=None, reduce_mean=False, continuous=True, likelihood_weighting=False, model_type="m"):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        tareget: String. If "auto", target is automatically select among ["noise", "gt_motion"].
        train: Bool.
        noise_sampler: A function to sample noise
        energy: please ignore this. If True, the diffusion model output the scaler instead vector. Then, vector is obtained by computing the gradient of scaler value respect to motion.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
        mdoel_type: output type

    Returns:
        A one-step function for training or evaluation.
    """

    if target == "auto":
        if model_type in ["m", "l"]:
            target = "noise"
        else:
            target = "gt_motion"

    if continuous:
        if model_type == "m":
            if noise_sampler != None:
                raise NotImplementedError()
            
            if target == "noise":
                print("LOSS: sde loss")

                def get_sde_loss_fn_CG(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
                    """Create a loss function for training with arbirary SDEs.

                    Args:
                        sde: An `sde_lib.SDE` object that represents the forward SDE.
                        train: `True` for training loss and `False` for evaluation loss.
                        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
                        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
                        ad-hoc interpolation to take continuous time steps.
                        likelihood_weighting: If `True`, weight the mixture of score matching losses
                        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
                        eps: A `float` number. The smallest time step to sample from.

                    Returns:
                        A loss function.
                    """
                    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

                    def loss_fn(model, batch, condition=None, retrieved=None):
                        """Compute the loss function.

                        Args:
                            model: A score model.
                            batch: A mini-batch of training data.

                        Returns:
                            loss: A scalar that represents the average loss value across the mini-batch.
                        """
                        score_fn = utils.get_score_fn_CG(sde, model, train=train, continuous=continuous)
                        temp_ins = batch[list(batch.keys())[0]]
                        B = temp_ins.shape[0]
                        device = temp_ins.device
                        t = torch.rand(B, device=device) * (sde.T - eps) + eps
                        mean, std = sde.marginal_prob(batch, t)
                        perturbed_data = {}
                        z = {}
                        for key in batch.keys():
                            z[key] = torch.randn_like(batch[key])
                            if batch[key].dim() == 2:
                                perturbed_data[key] = mean[key] + std[:, None] * z[key]
                            elif batch[key].dim() == 3:
                                perturbed_data[key] = mean[key] + std[:, None, None] * z[key]

                        score = score_fn(perturbed_data, t, condition, retrieved)

                        total_loss = 0
                        loss_dict = {}
                        for key in score.keys():
                            if not likelihood_weighting:
                                    if score[key].dim() == 2:
                                        losses = torch.square(score[key] * std[:, None] + z[key])
                                    elif score[key].dim() == 3:
                                        losses = torch.square(score[key] * std[:, None, None] + z[key])

                                    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
                            else:
                                zero_batch = {}
                                for temp_key in score.keys():
                                    zero_batch[temp_key] = torch.zeros_like(batch[temp_key])
                                
                                g2 = sde.sde(zero_batch, t)[1] ** 2
                                
                                if score[key].dim() == 2:
                                    losses = torch.square(score[key] + z[key] / std[:, None])
                                elif score[key].dim() == 3:
                                    losses = torch.square(score[key] + z[key] / std[:, None, None])

                                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

                            loss = torch.mean(losses)
                            total_loss += loss
                            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
                        
                        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
                        return total_loss, loss_dict

                    return loss_fn

                loss_fn = get_sde_loss_fn_CG(sde, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)

            elif target == "gt_motion":
                raise NotImplementedError()
                # print("LOSS: Cold Diffusion loss")
                # if likelihood_weighting==True:
                #     warnings.warn("Likelihood weighting is not supported for ColdDiffusion_like_loss", UserWarning)
                # loss_fn = get_motion_loss_fn(sde, train, reduce_mean=reduce_mean)
            
        elif model_type == "l":
            raise NotImplementedError()
            # if noise_sampler == None:
            #     raise NotImplementedError()
            
            # if target == "noise":
            #     print("LOSS: latent sde loss")
            #     loss_fn = get_latent_sde_loss_fn(sde, train, noise_sampler, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
            # elif target == "gt_motion":
            #     print("LOSS: Latent Cold Diffusion loss")
            #     if likelihood_weighting==True:
            #         warnings.warn("Likelihood weighting is not supported for ColdDiffusion_like_loss", UserWarning)
            #     loss_fn = get_latent_motion_loss_fn(sde, train, noise_sampler, reduce_mean=reduce_mean)

        elif model_type == "l-m":
            raise NotImplementedError()
            # if likelihood_weighting==True:
            #     warnings.warn("Likelihood weighting is not supported for ColdDiffusion_like_loss", UserWarning)
            
            # if target == "noise":
            #     raise ValueError("target should be gt_motion for l-m type model")
            # elif target == "gt_motion":
            #     print("LOSS: latent Cold Diffusion loss for l-m")
            #     loss_fn = get_latent_motion_loss_fn_for_lm(sde, train, noise_sampler, reduce_mean=reduce_mean)      
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE) and energy:
            raise NotImplementedError("TODO")
        elif isinstance(sde, VESDE) and not energy:
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE) and energy:
            loss_fn = get_energy_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE) and not energy:
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    return loss_fn

def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = utils.get_score_fn(sde, model, False, train=train, continuous=continuous)
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(batch, retrieved, t)
        else:
            mean, std = sde.marginal_prob(batch, t)
        perturbed_data = {}
        z = {}
        for key in batch.keys():
            z[key] = torch.randn_like(batch[key])
            if batch[key].dim() == 2:
                perturbed_data[key] = mean[key] + std[:, None] * z[key]
            elif batch[key].dim() == 3:
                perturbed_data[key] = mean[key] + std[:, None, None] * z[key]

        score = score_fn(perturbed_data, t, condition)

        total_loss = 0
        loss_dict = {}
        for key in score.keys():
            if not likelihood_weighting:
                    if score[key].dim() == 2:
                        losses = torch.square(score[key] * std[:, None] + z[key])
                    elif score[key].dim() == 3:
                        losses = torch.square(score[key] * std[:, None, None] + z[key])

                    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            else:
                zero_batch = {}
                for temp_key in score.keys():
                    zero_batch[temp_key] = torch.zeros_like(batch[temp_key])
                
                g2 = sde.sde(zero_batch, t)[1] ** 2
                
                if score[key].dim() == 2:
                    losses = torch.square(score[key] + z[key] / std[:, None])
                elif score[key].dim() == 3:
                    losses = torch.square(score[key] + z[key] / std[:, None, None])

                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_motion_loss_fn(sde, train, reduce_mean=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A model that predicts the gt_motion instead of score.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """

        # get batch size and device
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device

        # get model function (do model.train etc..)
        model_fn = utils.get_model_fn(model, train=train)

        # get perturbed data
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(batch, retrieved, t)
        else:
            mean, std = sde.marginal_prob(batch, t)
        perturbed_data = {}
        for key in batch.keys():
            noise = torch.randn_like(batch[key])
            if batch[key].dim() == 2:
                perturbed_data[key] = mean[key] + std[:, None] * noise
            elif batch[key].dim() == 3:
                perturbed_data[key] = mean[key] + std[:, None, None] * noise

        # prediction
        predicted_query = model_fn(perturbed_data, t, condition)

        total_loss = 0
        loss_dict = {}
        for key in predicted_query.keys():
            losses = torch.square(predicted_query[key] - batch[key])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_latent_sde_loss_fn(sde, train, noise_sampler, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """

        # get batch size and device
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device

        # get latent
        with torch.no_grad():
            latent = model.latent_AutoEncoder.encode(batch)
            if isinstance(sde, IRSDE):
                retrieved_latent = model.latent_AutoEncoder.encode(retrieved)

        score_fn = utils.get_score_fn(sde, model, False, train=train, continuous=continuous)
        
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(latent, retrieved_latent, t)
        else:
            mean, std = sde.marginal_prob(latent, t)
        perturbed_data = {}
        z = noise_sampler(latent)
        for key in latent.keys():
            perturbed_data[key] = mean[key] + std[:, None] * z[key]

        score = score_fn(perturbed_data, t, condition)

        total_loss = 0
        loss_dict = {}
        for key in score.keys():
            if not likelihood_weighting:
                losses = torch.square(score[key] * std[:, None] + z[key])
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            else:
                zero_batch = {}
                for temp_key in score.keys():
                    zero_batch[temp_key] = torch.zeros_like(latent[temp_key])
                
                g2 = sde.sde(zero_batch, t)[1] ** 2
                
                losses = torch.square(score[key] + z[key] / std[:, None])
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_latent_motion_loss_fn(sde, train, noise_sampler, reduce_mean=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """

        # get batch size and device
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device

        # get latent
        with torch.no_grad():
            latent = model.latent_AutoEncoder.encode(batch)
            if isinstance(sde, IRSDE):
                retrieved_latent = model.latent_AutoEncoder.encode(retrieved)

        # get model function (do model.train etc..)
        model_fn = utils.get_model_fn(model, train=train)
        
        # get perturbed data
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(latent, retrieved_latent, t)
        else:
            mean, std = sde.marginal_prob(latent, t)
        perturbed_data = {}
        z = noise_sampler(latent)
        for key in latent.keys():
            perturbed_data[key] = mean[key] + std[:, None] * z[key]

        # prediction
        predicted_latent = model_fn(perturbed_data, t, condition)

        total_loss = 0
        loss_dict = {}
        for key in predicted_latent.keys():
            losses = torch.square(predicted_latent[key] - latent[key])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_latent_motion_loss_fn_for_lm(sde, train, noise_sampler, reduce_mean=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        
        # get batch size and device
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device

        # get latent
        with torch.no_grad():
            latent = model.latent_AutoEncoder.encode(batch)
            if isinstance(sde, IRSDE):
                retrieved_latent = model.latent_AutoEncoder.encode(retrieved)

        # get model function (do model.train etc..)
        model_fn = utils.get_model_fn(model, train=train)
        
        # get perturbed data (latent)
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(latent, retrieved_latent, t)
        else:
            mean, std = sde.marginal_prob(latent, t)
        perturbed_data = {}
        z = noise_sampler(latent)
        for key in latent.keys():
            perturbed_data[key] = mean[key] + std[:, None] * z[key]

        # prediction. Note that output is not latent but motion
        predicted_query = model_fn(perturbed_data, t, condition)

        total_loss = 0
        loss_dict = {}
        for key in predicted_query.keys():
            losses = torch.square(predicted_query[key] - batch[key])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None):
        key_list = list(batch.keys())
        B = batch[key_list[0]].shape[0]
        device = batch[key_list[0]].device

        model_fn = utils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (B,), device=device)
        sigmas = smld_sigma_array.to(device)[labels]

        perturbed_data = {}
        noise = {}
        for key in batch.keys():
            if batch[key].dim() == 2:
                noise[key] = torch.randn_like(batch[key]) * sigmas[:, None]
            elif batch[key].dim() == 3:
                noise[key] = torch.randn_like(batch[key]) * sigmas[:, None, None]
            perturbed_data[key] = noise[key] + batch[key]

        score = model_fn(perturbed_data, labels, condition=condition)

        total_loss = 0
        loss_dict = {}
        for key in score.keys():
            if score[key].dim() == 2:
                target = -noise[key] / (sigmas ** 2)[:, None]
            elif score[key].dim() == 3:
                target = -noise[key] / (sigmas ** 2)[:, None, None]

            losses = torch.square(score[key] - target)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
            
            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()

        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None):
        key_list = list(batch.keys())
        B = batch[key_list[0]].shape[0]
        device = batch[key_list[0]].device

        model_fn = utils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (B,), device=device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(device)

        perturbed_data = {}
        noise = {}
        for key in batch.keys():
            noise[key] = torch.randn_like(batch[key])
            if batch[key].dim() == 2:
                perturbed_data[key] = sqrt_alphas_cumprod[labels, None] * batch[key] + \
                            sqrt_1m_alphas_cumprod[labels, None] * noise[key]
            elif batch[key].dim() == 3:
                perturbed_data[key] = sqrt_alphas_cumprod[labels, None, None] * batch[key] + \
                            sqrt_1m_alphas_cumprod[labels, None, None] * noise[key]

        score = model_fn(perturbed_data, labels, condition=condition)

        total_loss = 0
        loss_dict = {}
        for key in score.keys():
            losses = torch.square(score[key] - noise[key])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)

            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

####### Retrieval Loss #######
def get_triplet_loss_fn(train, margin1=0.5, margin2=2.0):

    def triplet_loss_fn(model, target_image, target_query, pos_image, pos_query, neg_image, neg_query):
        triplet_loss1 = torch.nn.TripletMarginLoss(margin=margin1, p=2)
        triplet_loss2 = torch.nn.TripletMarginLoss(margin=margin2, p=2)

        feature_list = []
        feature_list.append(model.get_extracted_feature(target_image, target_query)) # B S D
        feature_list.append(model.get_extracted_feature(pos_image, pos_query))
        feature_list.append(model.get_extracted_feature(target_image, pos_query))
        feature_list.append(model.get_extracted_feature(target_image, neg_query))

        loss_dict = {}
        feature_list = [rearrange(feature, "B S D -> B (S D)") for feature in feature_list]
        loss = triplet_loss1(feature_list[0], feature_list[1], feature_list[2]) + triplet_loss2(feature_list[0], feature_list[2], feature_list[3])

        loss_dict[f"{'train' if train else 'val'}/loss"] = loss.item()
        return loss, loss_dict

    return triplet_loss_fn

##################################################################################################################################################

# miscellaneous

def get_energy_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None):
        key_list = list(batch.keys())
        B = batch[key_list[0]].shape[0]
        device = batch[key_list[0]].device

        model_fn = utils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (B,), device=device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(device)

        perturbed_data = {}
        noise = {}
        for key in batch.keys():
            noise[key] = torch.randn_like(batch[key])
            if batch[key].dim() == 2:
                perturbed_data[key] = sqrt_alphas_cumprod[labels, None] * batch[key] + \
                            sqrt_1m_alphas_cumprod[labels, None] * noise[key]
                perturbed_data[key] = perturbed_data[key].detach()
                perturbed_data[key].requires_grad_(True)
            elif batch[key].dim() == 3:
                perturbed_data[key] = sqrt_alphas_cumprod[labels, None, None] * batch[key] + \
                            sqrt_1m_alphas_cumprod[labels, None, None] * noise[key]
                perturbed_data[key] = perturbed_data[key].detach()
                perturbed_data[key].requires_grad_(True)

        # pred energy 
        energy = model_fn(perturbed_data, labels, condition=condition)

        # get score via differentiation of energy with respect to perturbed_data
        sum_energy = energy.sum()
        grad_x = torch.autograd.grad(sum_energy, [perturbed_data[key] for key in perturbed_data.keys()], create_graph=True)
        # sum_energy.backward(create_graph=True)
        
        total_loss = 0
        loss_dict = {}
        for key, grad in zip(perturbed_data.keys(), grad_x):
            losses = torch.square(grad - noise[key])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)

            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn

def get_latent_motion_sde_loss_fn(sde, train, reduce_mean=True, pred_type="m0", eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """

        # get batch size and device
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device

        # get latent
        with torch.no_grad():
            latent = model.latent_AutoEncoder.encode(batch)
            if isinstance(sde, IRSDE):
                retrieved_latent = model.latent_AutoEncoder.encode(retrieved)
        
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(latent, retrieved_latent, t)
        else:
            mean, std = sde.marginal_prob(latent, t)

        perturbed_data = {}
        z = {}
        for key in latent.keys():
            z[key] = torch.randn_like(latent[key])
            perturbed_data[key] = mean[key] + std[:, None] * z[key]

        total_loss = 0
        loss_dict = {}

        if pred_type == "m0":
            # get model function
            model_fn = utils.get_model_fn(model, train=train)
            
            # prediction. Note that output is not latent but motion
            predicted_query = model_fn(perturbed_data, t, condition)

            for key in predicted_query.keys():
                losses = torch.square(predicted_query[key] - batch[key])
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
                loss = torch.mean(losses)
                total_loss += loss
                loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        elif pred_type == "mt":
            # get model function
            model_fn = utils.get_model_fn(model, train=train)

            # prediction. Note that output is not latent but motion
            predicted_query = model_fn(perturbed_data, t, condition)

            # get target
            mt = model.latent_AutoEncoder.encode(mean)

            for key in predicted_query.keys():
                losses = torch.square(predicted_query[key] - mt[key])
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
                loss = torch.mean(losses)
                total_loss += loss
                loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        elif pred_type == "zt":
            raise NotImplementedError("TODO")
            score_fn = utils.get_score_fn(sde, model, False, train=train, continuous=True)
            score = score_fn(perturbed_data, t, condition)

            for key in score.keys():
                losses = torch.square(score[key] * std[:, None] + z[key])
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

                loss = torch.mean(losses)
                total_loss += loss
                loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        else:
            raise ValueError(f"Invalid pred type {pred_type}.")

        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict      

    return loss_fn

def get_latent_IRSDE_loss_fn(sde, train, noise_sampler, reduce_mean=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition=None, retrieved=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        
        # get batch size and device
        temp_ins = batch[list(batch.keys())[0]]
        B = temp_ins.shape[0]
        device = temp_ins.device

        # get latent
        with torch.no_grad():
            latent = model.latent_AutoEncoder.encode(batch)
            retrieved_latent = model.latent_AutoEncoder.encode(retrieved)

        # get model function
        model_fn = utils.get_model_fn(model, train=train)
        
        # get perturbed data (latent)
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        if isinstance(sde, IRSDE):
            mean, std = sde.marginal_prob(latent, retrieved_latent, t)
        else:
            mean, std = sde.marginal_prob(latent, t)
        perturbed_data = {}
        # z = {}
        z = noise_sampler(latent)
        for key in latent.keys():
            # z[key] = torch.randn_like(latent[key])
            perturbed_data[key] = mean[key] + std[:, None] * z[key]

        # prediction. Note that output is not latent but motion
        predicted_query = model_fn(perturbed_data, t, condition)

        total_loss = 0
        loss_dict = {}
        for key in predicted_query.keys():
            losses = torch.square(predicted_query[key] - batch[key])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
            total_loss += loss
            loss_dict[f"{'train' if train else 'val'}/{key}"] = loss.item()
        
        loss_dict[f"{'train' if train else 'val'}/loss"] = total_loss.item()
        return total_loss, loss_dict

    return loss_fn