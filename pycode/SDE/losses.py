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

import torch
import torch.optim as optim
import numpy as np
from . import utils
from .sde_lib import VESDE, VPSDE

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

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = utils.get_score_fn(sde, model, train=train, continuous=continuous)
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

        score = score_fn(perturbed_data, t)

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

def get_loss_fn(sde, train, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
        A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    return loss_fn