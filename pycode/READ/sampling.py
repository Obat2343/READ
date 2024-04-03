# This code is based on the original sde paper and modified to change inputs from image to motion.
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

"""Various sampling methods."""

import functools

import copy
import torch
import numpy as np
import abc

from scipy import integrate

from . import utils
from ..READ import sde_lib

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]

def get_sampling_fn(cfg, sde, shape, inverse_scaler, eps, device="cuda", noise_sampler=None, guidance=False):
    """Create a sampling function.

    Args:
        config: A `CfgNode` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
        noise_sampler: A function to sample noise
    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    sampler_name = cfg.SDE.SAMPLING.METHOD
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                    shape_dict=shape,
                                    inverse_scaler=inverse_scaler,
                                    denoise=cfg.SDE.SAMPLING.NOISE_REMOVAL,
                                    eps=eps,
                                    device=device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(cfg.SDE.SAMPLING.PREDICTOR.lower())
        corrector = get_corrector(cfg.SDE.SAMPLING.CORRECTOR.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                    shape_dict=shape,
                                    predictor=predictor,
                                    corrector=corrector,
                                    inverse_scaler=inverse_scaler,
                                    snr=cfg.SDE.SAMPLING.SNR,
                                    noise_sampler=noise_sampler,
                                    n_steps=cfg.SDE.SAMPLING.N_STEPS_EACH,
                                    probability_flow=cfg.SDE.SAMPLING.PROBABILITY_FLOW,
                                    continuous=cfg.SDE.TRAINING.CONTINUOUS,
                                    model_inout_mode=cfg.MODEL.INOUT,
                                    denoise=cfg.SDE.SAMPLING.NOISE_REMOVAL,
                                    guidance=guidance,
                                    eps=eps,
                                    device=device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False, noise_sampler=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn
        self.noise_sampler = noise_sampler

    @abc.abstractmethod
    def update_fn(self, x, t, dt, condition=None, retrieved=None):
        """One update of the predictor.

        Args:
        x: A PyTorch tensor representing the current state
        t: A Pytorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, condition=None):
        """One update of the corrector.

        Args:
        x: A PyTorch tensor representing the current state
        t: A PyTorch tensor representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, dt, condition=None, retrieved=None):
        # dt = -1. / self.rsde.N

        x_mean = {}
        drift, diffusion = self.rsde.sde(x, t, condition, retrieved)
        
        for key in drift.keys():
            z = torch.randn_like(x[key])
            x_mean[key] = x[key] + drift[key] * dt
            if x_mean[key].dim() == 2:
                x[key] = x_mean[key] + diffusion[:, None] * torch.sqrt(-dt) * z
            elif x_mean[key].dim() == 3:
                x[key] = x_mean[key] + diffusion[:, None, None] * torch.sqrt(-dt) * z
        return x, x_mean

@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, dt, condition=None, retrieved=None):
        f, G = self.rsde.discretize(x, t, condition, retrieved)
        x_mean = {}
        for key in f.keys():
            z = torch.randn_like(x[key])
            x_mean[key] = x[key] - f[key]
            if x_mean[key].dim() == 2:
                x[key] = x_mean[key] + G[:, None] * z
            elif x_mean[key].dim() == 3:
                x[key] = x_mean[key] + G[:, None, None] * z
        return x, x_mean

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t, condition=None):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t, condition)
        x_mean = {}
        for key in score.keys():
            if score[key].dim() == 2:
                x_mean[key] = x[key] + score[key] * (sigma ** 2 - adjacent_sigma ** 2)[:, None]
            elif score[key].dim() == 3:
                x_mean[key] = x[key] + score[key] * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None]
            
            std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
            noise = torch.randn_like(x[key])

            if x_mean[key].dim() == 2:
                x[key] = x_mean[key] + std[:, None] * noise
            elif x_mean[key].dim() == 3:
                x[key] = x_mean[key] + std[:, None, None] * noise

        return x, x_mean

    def vpsde_update_fn(self, x, t, condition=None, retrieved=None):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t, condition)
        x_mean = {}
        for key in score.keys():
            noise = torch.randn_like(x[key])
            if score[key].dim() == 2:
                x_mean[key] = (x[key] + beta[:, None] * score[key]) / torch.sqrt(1. - beta)[:, None]
                x[key] = x_mean[key] + torch.sqrt(beta)[:, None] * noise
            elif score[key].dim() == 3:
                x_mean[key] = (x[key] + beta[:, None, None] * score[key]) / torch.sqrt(1. - beta)[:, None, None]
                x[key] = x_mean[key] + torch.sqrt(beta)[:, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t, dt, condition=None):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t, condition)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t, condition)

@register_predictor(name='cold_diffusion')
class ColdDiffusionPredictor(Predictor):
    def __init__(self, sde, model_fn, model, noise_sampler, probability_flow=False, inout_mode="l-m"):
        super().__init__(sde, None, probability_flow, noise_sampler=noise_sampler)
        self.model_fn = model_fn
        self.model = model
        self.inout_mode = inout_mode

    def update_fn(self, x, t, dt, condition=None, retrieved=None):
        # dt = -1. / self.rsde.N
        # t_next = torch.clamp(t + dt, min=0.)
        t_next = t + dt

        # encode
        if self.inout_mode != "m":
            input_motion = self.model.latent_AutoEncoder.encode(x)
        else:
            input_motion = copy.deepcopy(x)
        
        # prediction
        x_mean = self.model_fn(input_motion, t, condition)
        
        # add noise by forward SDE
        if self.inout_mode != "m":
            latent = self.model.latent_AutoEncoder.encode(x_mean)
            if isinstance(self.sde, sde_lib.IRSDE):
                retrieved_latent = self.model.latent_AutoEncoder.encode(retrieved)
                mean, std = self.sde.marginal_prob(latent, retrieved_latent, t_next)
            else:
                mean, std = self.sde.marginal_prob(latent, t_next)

            perturbed_latent = {}
            z = self.noise_sampler(latent)
            for key in latent.keys():
                perturbed_latent[key] = mean[key] + std[:, None] * z[key]
            
            x = self.model.latent_AutoEncoder.decode(perturbed_latent)

        else:
            if isinstance(self.sde, sde_lib.IRSDE):
                mean, std = self.sde.marginal_prob(x_mean, retrieved, t_next)
            else:
                mean, std = self.sde.marginal_prob(x_mean, t_next)

            x = {}
            for key in mean.keys():
                x[key] = mean[key] + std[:, None, None] * torch.randn_like(mean[key])

        return x, x_mean
    
@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, condition=None):
        return x, x

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, condition=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t, condition)
            x_mean = {}
            for key in grad.keys():
                noise = torch.randn_like(x[key])
                grad_norm = torch.norm(grad[key].reshape(grad[key].shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                if grad[key].dim() == 2:
                    x_mean[key] = x[key] + step_size[:, None] * grad[key]
                    x[key] = x_mean[key] + torch.sqrt(step_size * 2)[:, None] * noise
                elif grad[key].dim() == 3:
                    x_mean[key] = x[key] + step_size[:, None, None] * grad[key]
                    x[key] = x_mean[key] + torch.sqrt(step_size * 2)[:, None, None] * noise

        return x, x_mean

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, condition=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t, condition)
            x_mean = {}
            for key in grad.keys():
                noise = torch.randn_like(x[key])
                step_size = (target_snr * std) ** 2 * 2 * alpha
                if grad[key].dim() == 2:
                    x_mean[key] = x[key] + step_size[:, None] * grad[key]
                    x[key] = x_mean[key] + noise * torch.sqrt(step_size * 2)[:, None]
                elif grad[key].dim() == 3:
                    x_mean[key] = x[key] + step_size[:, None, None] * grad[key]
                    x[key] = x_mean[key] + noise * torch.sqrt(step_size * 2)[:, None, None]

        return x, x_mean

@register_corrector(name='gd')
class GradientDescent(Corrector):
    """
    AnnealedLangevinDynamics without noise.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, condition=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t, condition)
            x_mean = {}
            for key in grad.keys():
                noise = torch.randn_like(x[key])
                step_size = (target_snr * std) ** 2 * 2 * alpha
                if grad[key].dim() == 2:
                    x_mean[key] = x[key] + step_size[:, None] * grad[key]
                    x[key] = x_mean[key]
                elif grad[key].dim() == 3:
                    x_mean[key] = x[key] + step_size[:, None, None] * grad[key]
                    x[key] = x_mean[key]

        return x, x_mean

@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, condition=None):
        return x, x

def shared_predictor_update_fn(x, t, dt, sde, model, predictor, probability_flow, continuous, condition=None, retrieved=None, noise_sampler=None, mode="m", guidance=False):
    """A wrapper that configures and returns the update function of predictors."""
    if type(condition) == type(None):
        if guidance:
            score_fn = utils.get_score_fn_CG(sde, model, train=False, continuous=continuous, fixed_condition=False)
        else:
            score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous)
    else:
        if guidance:
            score_fn = utils.get_score_fn_CG(sde, model, train=False, continuous=continuous, fixed_condition=True)
        else:
            score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous, fixed_condition=True)

    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    elif predictor is ColdDiffusionPredictor:
        model_fn = utils.get_model_fn(model, train=False)
        if probability_flow == True:
            raise NotImplementedError()
        predictor_obj = predictor(sde, model_fn, model, noise_sampler, False, inout_mode=mode)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, dt, condition, retrieved)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps, condition=None, guidance=False):
    """A wrapper tha configures and returns the update function of correctors."""
    if type(condition) == type(None):
        score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous)
    else:
        score_fn = utils.get_score_fn(sde, model, train=False, continuous=continuous, fixed_condition=True)

    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, condition)

def get_pc_sampler(sde, shape_dict, predictor, corrector, inverse_scaler, snr, noise_sampler=None,
                    n_steps=1, probability_flow=False, continuous=False, model_inout_mode="l-m",
                    denoise=True, eps=1e-3, device='cuda', guidance=False):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        sde: An `sde_lib.SDE` object representing the forward SDE.
        shape_dict: A dict of shapes.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            noise_sampler=noise_sampler,
                                            mode=model_inout_mode,
                                            guidance=guidance)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(model, x=None, condition=None, start_time_ratio=None, N=None, history=False):
        """ The PC sampler funciton.

        Args:
        model: A score model.
        x: If present, generate samples from x. x is retrieved motion normally.
        start_time_ratio=None: If present, the generation starts from sde.T * start_time_ratio
        N: If present, the number of iteration of updating x is N
        latent: If input is latent, please use True
        Returns:
        Samples, number of function evaluations.
        """
        with torch.no_grad():
            # set timestep
            if start_time_ratio == None:
                start_time = sde.T
            else:
                start_time = sde.T * start_time_ratio
            print(f"start time: {start_time}")
            
            if N == None:
                N = sde.N
            else:
                sde.N = N
            
            # Initial sample
            if type(x) == type(None):
                x = sde.prior_sampling(shape_dict)
                x_type = "random"
            else:
                x = copy.deepcopy(x)
                x_type = "retrieved"
                print("use retrieved data for sampling")
            
            # to device
            for key in x.keys():
                x[key] = x[key].to(device)
            B = x[key].shape[0]
            
            # pre-compute the conditioning feature
            if type(condition) != type(None):
                condition = condition.to(device)
                model.set_condition(condition)
            
            # set initial motion
            if x_type == "random":
                retrieved = None
                x_mean = {}
                for key in x.keys():
                    x_mean[key] = torch.zeros_like(x[key])

                if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.VESDE):

                    if model_inout_mode in ["m", "l"]:
                        pass
                    elif model_inout_mode in ["l-m"]:
                        raise NotImplementedError("TODO")
                    else:
                        raise ValueError("Invalid inout mode")
                    
                elif isinstance(sde, sde_lib.IRSDE):
                    raise ValueError("Initial sample should be sampled based on a retrieved motion.")

                else:
                    raise ValueError("Invalid sde")
                
            else:
                retrieved = copy.deepcopy(x)
                x_mean = copy.deepcopy(x)
                print(f"reverse process starts from {start_time}")
                if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.VESDE):

                    if model_inout_mode == "m":
                        mean, std = sde.marginal_prob(x, torch.ones(B, device=device) * start_time)
                        for key in x.keys():
                            x[key] = mean[key] + std[:, None, None] * torch.randn_like(x[key])

                    elif model_inout_mode == "l":
                        raise NotImplementedError("Not Debugged")
                        z = model.latent_AutoEncoder.encode(x)
                        mean, std = sde.marginal_prob(z, torch.ones(B, device=device) * start_time)
                        x = {}
                        for key in mean.keys():
                            x[key] = mean[key] + std[:, None] * torch.randn_like(z[key])

                    elif model_inout_mode == "l-m":
                        raise NotImplementedError("TODO")
                    
                elif isinstance(sde, sde_lib.IRSDE):

                    if model_inout_mode == "m":
                        mean, std = sde.marginal_prob(x, retrieved, torch.ones(B, device=device) * start_time)
                        for key in x.keys():
                            x[key] = retrieved[key] + sde.lamda * torch.randn_like(x[key])

                    elif model_inout_mode == "l":
                        retrieved = model.latent_AutoEncoder.encode(x)
                        mean, std = sde.marginal_prob(retrieved, retrieved, torch.ones(B, device=device) * start_time)
                        x = {}
                        for key in mean.keys():
                            x[key] = mean[key] + std[:, None] * torch.randn_like(retrieved[key])
                    
                    elif model_inout_mode == "l-m":
                        # implementation option: 1. directly use x0, 2. add noise
                        pass # Note: Is this good way?
                    
                    else:
                        raise ValueError("Invalid inout mode")

                else:
                    raise ValueError("Invalid sde")

            # set time-step
            timesteps = torch.linspace(start_time, eps, N+1, device=device)

            # timesteps = torch.linspace(start_time, 0.1, int(N/2), device=device)
            # additional_step = torch.linspace(0.1, eps, int(N/2)+2, device=device)
            # timesteps = torch.cat([timesteps, additional_step[1:]], 0)
            
            # save initial motion
            if history:
                history = {}
                history["x"], history["x_mean"] = [], []

                if model_inout_mode == "l":
                    history["x"].append(copy.deepcopy(model.latent_AutoEncoder.decode(x)))
                    history["x_mean"].append(copy.deepcopy(inverse_scaler(model.latent_AutoEncoder.decode(mean))))
                else:
                    history["x"].append(copy.deepcopy(inverse_scaler(x)))
                    history["x_mean"].append(copy.deepcopy(inverse_scaler(x_mean)))
            
            # start reverse sde
            for i in range(N):
                print(f"{i+1}/{N}")
                t = timesteps[i]
                dt = timesteps[i+1] - timesteps[i]
                vec_t = torch.ones(B, device=t.device) * t
                vec_dt = torch.ones(B, device=t.device) * dt
                x, x_mean = corrector_update_fn(x, vec_t, model=model, condition=condition)
                x, x_mean = predictor_update_fn(x, vec_t, vec_dt, model=model, condition=condition, retrieved=retrieved)
                
                if history:
                    if model_inout_mode == "l":
                        history["x"].insert(0, copy.deepcopy(model.latent_AutoEncoder.decode(x)))
                        history["x_mean"].insert(0, copy.deepcopy(inverse_scaler(model.latent_AutoEncoder.decode(x_mean))))
                    else:
                        history["x"].insert(0, copy.deepcopy(inverse_scaler(x)))
                        history["x_mean"].insert(0, copy.deepcopy(inverse_scaler(x_mean)))
            
            if model_inout_mode == "l":
                output = inverse_scaler(model.latent_AutoEncoder.decode(x_mean if denoise else x))
            else:
                output = inverse_scaler(x_mean if denoise else x)

            if history:
                return output, history, N * (n_steps + 1)
            else:
                return output, N * (n_steps + 1)

    return pc_sampler

def get_ode_sampler(sde, shape_dict, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape_dict: A dict of shapes.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, condition=None, retrieved=None, CG=False):
        if type(condition) == type(None):
            if CG:
                score_fn = utils.get_score_fn_CG(sde, model, train=False, continuous=True)
            else:
                score_fn = utils.get_score_fn(sde, model, train=False, continuous=True)
        else:
            if CG:
                score_fn = utils.get_score_fn_CG(sde, model, train=False, continuous=True, fixed_condition=True)
            else:
                score_fn = utils.get_score_fn(sde, model, train=False, continuous=True, fixed_condition=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        for key in x.keys():
            B = x[key].shape[0]
            device = x[key].device
            break

        vec_eps = torch.ones(B, device=device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, None, condition=condition, retrieved=retrieved)
        return x

    def drift_fn(model, x, t, condition=None, retrieved=None, fixed_condition=False, CG=False):
        """Get the drift function of the reverse-time SDE."""
        if CG:
            score_fn = utils.get_score_fn_CG(sde, model, train=False, continuous=True, fixed_condition=fixed_condition)
        else:
            score_fn = utils.get_score_fn(sde, model, train=False, continuous=True, fixed_condition=fixed_condition)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, condition, retrieved)[0]

    def ode_sampler(model, x=None, condition=None, history=False, start_time_ratio=None, N=None, CG=False):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
        model: A score model.
        x: If present, generate samples from latent code `x`.
        Returns:
        samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if type(x) == type(None):
                # If) not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape_dict)
                retrieved = None
            elif CG == True:
                retrieved = copy.deepcopy(x)
                for key in retrieved.keys():
                    retrieved[key] = retrieved[key].to(device)
                x = sde.prior_sampling(shape_dict)
            else:
                raise NotImplementedError()
            
            if type(condition) != type(None):
                condition = condition.to(device)
                model.set_condition(condition)

            # Change device
            for key in x.keys():
                x[key] = x[key].to(device)

            # prepare function converting a dict to a numpy array
            key_list = list(shape_dict.keys())
            key_list.sort()
            B = shape_dict[key_list[0]][0]
            
            from_flattened_numpy = functools.partial(utils.from_flattened_numpy, key_list=key_list, shape_dict=shape_dict)
            to_flattened_numpy = functools.partial(utils.to_flattened_numpy, key_list=key_list)

            # define ode func
            def ode_func(t, x):
                x = from_flattened_numpy(x)
                for key in x.keys():
                    x[key] = x[key].to(device).type(torch.float32)
                vec_t = torch.ones(B, device=device) * t
                if type(condition) == type(None):
                    drift = drift_fn(model, x, vec_t, condition, retrieved, CG=CG)
                else:
                    drift = drift_fn(model, x, vec_t, condition, retrieved, fixed_condition=True, CG=CG)
                return to_flattened_numpy(drift)

            # define start step
            if start_time_ratio == None:
                start_time = sde.T
            else:
                start_time = sde.T * start_time_ratio

            # 
            if N != None:
                print("WARNING: ODE solvers automatically compute N")

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (start_time, eps), to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = from_flattened_numpy(solution.y[:, -1])
            for key in x.keys():
                x[key] = x[key].to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, condition=condition, retrieved=retrieved, CG=CG)

            # denorm x
            x = inverse_scaler(x)
            return x, None, nfe   

    return ode_sampler