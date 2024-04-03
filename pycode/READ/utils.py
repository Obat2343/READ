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
import warnings
from ..READ import sde_lib

def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, labels, condition=None, fixed_condition=False):
        """Compute the output of the score-based model.

        Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
        condition: A mini-batch of condition.
        fixed_condition: if `True`, the model uses the preset condition. 

        Returns:
        A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
        else:
            model.train()

        # pred noise (not score)
        if fixed_condition:
            noise = model(None, x, labels, with_feature=True)
        elif type(condition) == type(None):
            noise = model(x, labels)
        else:
            noise = model(condition, x, labels)

        return noise

    return model_fn

def get_score_fn(sde, model, train=False, continuous=False, fixed_condition=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.
        fixed_condition: if `True`, the model uses the preset condition.
    Returns:
        A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if (isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE)):
        def score_fn(x, t, condition=None, retrieved=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels, condition, fixed_condition)
                zero_x = {}
                for key in x.keys():
                    zero_x[key] = torch.zeros_like(x[key])
                std = sde.marginal_prob(zero_x, t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels, condition, fixed_condition)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            for key in score.keys():
                if score[key].dim() == 2:
                    score[key] = -score[key] / std[:, None]
                elif score[key].dim() == 3:
                    score[key] = -score[key] / std[:, None, None]
            return score
    
    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t, condition=None, retrieved=None):
            if continuous:
                zero_x = {}
                for key in x.keys():
                    zero_x[key] = torch.zeros_like(x[key])
                labels = sde.marginal_prob(zero_x, t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels, condition, fixed_condition)
            return score

    elif isinstance(sde, sde_lib.IRSDE):
        def score_fn(x, t, condition=None, retrieved=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # continuously-trained models.
                score = model_fn(x, t, condition, fixed_condition)
                zero_x = {}
                for key in x.keys():
                    zero_x[key] = torch.zeros_like(x[key])
                std = sde.marginal_prob(zero_x, zero_x, t)[1]
            else:
                raise NotImplementedError("TODO")

            for key in score.keys():
                if score[key].dim() == 2:
                    score[key] = -score[key] / std[:, None]
                elif score[key].dim() == 3:
                    score[key] = -score[key] / std[:, None, None]
            return score
        # warnings.warn(f"\nSDE class {sde.__class__.__name__} is not yet debugged.", UserWarning)
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn

def to_flattened_numpy(dict, key_list):
    array = []
    for key in key_list:
        array.append(dict[key])
    D = dict[key].dim()

    array = torch.cat(array, D-1)
    return array.detach().cpu().numpy().reshape((-1,))
    
def from_flattened_numpy(array, key_list, shape_dict):
    last_dim = 0
    for key in key_list:
        shape = shape_dict[key]
        if len(shape) == 2:
            B, D = shape
        elif len(shape) == 3:
            B, S, D = shape
        last_dim += D
    
    if len(shape) == 2:
        total_shape = (B, last_dim)
    else:
        total_shape = (B, S, last_dim)
    
    array = torch.from_numpy(array.reshape(total_shape))
    current_dim = 0
    query = {}
    for i, key in enumerate(key_list):
        if len(shape) == 2:
            query[key] = array[:,current_dim:current_dim+shape_dict[key][len(shape)-1]]
        elif len(shape) == 3:
            query[key] = array[:,:,current_dim:current_dim+shape_dict[key][len(shape)-1]]

        current_dim += shape_dict[key][len(shape)-1]
    return query

def get_model_fn_CG(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, labels, condition, retrieval, fixed_condition=False):
        """Compute the output of the score-based model.

        Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
        condition: A mini-batch of condition.
        fixed_condition: if `True`, the model uses the preset condition. 

        Returns:
        A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            # pred noise (not score)
            if fixed_condition:
                noise = model.classifier_free_guidance(None, x, retrieval, labels, with_feature=True, guidance_rate=1.0)
            else:
                print("cfg")
                noise = model.classifier_free_guidance(condition, x, retrieval, labels, guidance_rate=1.0)
        else:
            model.train()
            # pred noise (not score)
            if fixed_condition:
                noise = model(None, x, retrieval, labels, with_feature=True)
            else:
                noise = model(condition, x, retrieval, labels)

        return noise

    return model_fn


def get_score_fn_CG(sde, model, train=False, continuous=False, fixed_condition=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.
        fixed_condition: if `True`, the model uses the preset condition.
    Returns:
        A score function.
    """
    model_fn = get_model_fn_CG(model, train=train)

    if (isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE)):
        def score_fn(x, t, condition=None, retrieved=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels, condition, retrieved, fixed_condition)
                zero_x = {}
                for key in x.keys():
                    zero_x[key] = torch.zeros_like(x[key])
                std = sde.marginal_prob(zero_x, t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels, condition, retrieved, fixed_condition)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            for key in score.keys():
                if score[key].dim() == 2:
                    score[key] = -score[key] / std[:, None]
                elif score[key].dim() == 3:
                    score[key] = -score[key] / std[:, None, None]
            return score

    elif isinstance(sde, sde_lib.IRSDE):
        def score_fn(x, t, condition=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # continuously-trained models.
                score = model_fn(x, t, condition, fixed_condition)
                zero_x = {}
                for key in x.keys():
                    zero_x[key] = torch.zeros_like(x[key])
                std = sde.marginal_prob(zero_x, zero_x, t)[1]
            else:
                raise NotImplementedError("TODO")

            for key in score.keys():
                if score[key].dim() == 2:
                    score[key] = -score[key] / std[:, None]
                elif score[key].dim() == 3:
                    score[key] = -score[key] / std[:, None, None]
            return score
        # warnings.warn(f"\nSDE class {sde.__class__.__name__} is not yet debugged.", UserWarning)
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn