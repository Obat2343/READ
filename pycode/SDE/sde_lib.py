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

"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""

import abc
import torch
import numpy as np

class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
        N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
        z: latent code
        Returns:
        log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
        x: a dict of torch tensors
        t: a torch float representing the diffusion time step (from 0 to `self.T`)

        Returns:
        f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = {}
        for key in drift.keys():
            f[key] = drift[key] * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
        score_fn: A time-dependent score-based model that takes x and t and returns the score.
        probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, condition=None):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift_dict, diffusion = sde_fn(x, t)
                score_dict = score_fn(x, t, condition)

                for key in score_dict.keys():
                    if score_dict[key].dim() == 2:
                        drift_dict[key] = drift_dict[key] - diffusion[:, None] ** 2 * score_dict[key] * (0.5 if self.probability_flow else 1.)
                    elif score_dict[key].dim() == 3:
                        drift_dict[key] = drift_dict[key] - diffusion[:, None, None] ** 2 * score_dict[key] * (0.5 if self.probability_flow else 1.)
                    else:
                        raise NotImplementedError()
                    
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift_dict, diffusion

            def discretize(self, x, t, condition=None):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f_dict, G = discretize_fn(x, t)
                rev_f_dict = {}
                score_dict = score_fn(x, t, condition)
                for key in score_dict.keys():
                    if score_dict[key].dim() == 2:
                        rev_f_dict[key] = f_dict[key] - G[:, None] ** 2 * score_dict[key] * (0.5 if self.probability_flow else 1.)
                    if score_dict[key].dim() == 3:
                        rev_f_dict[key] = f_dict[key] - G[:, None, None] ** 2 * score_dict[key] * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f_dict, rev_G

        return RSDE()

class VPSDE(SDE):
    def __init__(self, keys, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.keys = keys

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Args:
            x: dict of torch.tensor. input of the model
            t: float. diffusion step
        """
        drift_dict = {}
        for key in x.keys():
            beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
            if x[key].dim() == 2:
                drift_dict[key] = -0.5 * beta_t[:, None] * x[key]
            if x[key].dim() == 3:
                drift_dict[key] = -0.5 * beta_t[:, None, None] * x[key]

        diffusion = torch.sqrt(beta_t)
        return drift_dict, diffusion

    def marginal_prob(self, x, t):
        """
        Args:
            x: dict of torch.tensor. input of the model
            t: float. diffusion step
        """
        mean_dict = {}

        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        for key in x.keys():
            if x[key].dim() == 2:
                mean_dict[key] = torch.exp(log_mean_coeff[:, None]) * x[key]
            elif x[key].dim() == 3:
                mean_dict[key] = torch.exp(log_mean_coeff[:, None, None]) * x[key]
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean_dict, std

    def prior_sampling(self, shape_dict):
        prior_dict = {}
        for key in shape_dict.keys():
            prior_dict[key] = torch.randn(*shape_dict[key])
        return prior_dict

    def prior_logp(self, z):
        """
        Args:
            z: dict of torch.tensor.
        """
        D = z[self.keys[0]].dim()
        z = torch.cat([z[key] for key in self.keys], D-1)
        shape = z.shape
        N = np.prod(shape[1:])
        if D == 2:
            logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1)) / 2.
        elif D == 3:
            logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        device = x[self.keys[0]].device
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(device)[timestep]
        alpha = self.alphas.to(device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f_dict = {}
        for key in x.keys():
            if x[key].dim() == 2:
                f_dict[key] = torch.sqrt(alpha)[:, None] * x[key] - x[key]
            elif x[key].dim() == 3:
                f_dict[key] = torch.sqrt(alpha)[:, None, None] * x[key] - x[key]
        G = sqrt_beta
        return f_dict, G

class subVPSDE(SDE):
    def __init__(self, keys, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.keys = keys

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Args:
            x: dict of torch.tensor. input of the model
            t: float. diffusion step
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift_dict= {}
        for key in x.keys():
            if x[key].dim() == 2:
                drift_dict[key] = -0.5 * beta_t[:, None] * x[key]
            elif x[key].dim() == 3:
                drift_dict[key] = -0.5 * beta_t[:, None, None] * x[key]
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift_dict, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean_dict = {}
        for key in x.keys():
            if x[key].dim() == 2:
                mean_dict[key] = torch.exp(log_mean_coeff)[:, None] * x[key]
            elif x[key].dim() == 3:
                mean_dict[key] = torch.exp(log_mean_coeff)[:, None, None] * x[key]
        
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean_dict, std

    def prior_sampling(self, shape_dict):
        prior_dict = {}
        for key in shape_dict.keys():
            prior_dict[key] = torch.randn(*shape_dict[key])
        return prior_dict

    def prior_logp(self, z):    
        """
        Args:
            z: dict of torch.tensor.
        """
        D = z[self.keys[0]].dim()
        z = torch.cat([z[key] for key in self.keys], D-1)
        shape = z.shape
        N = np.prod(shape[1:])
        if D == 2:
            logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1,)) / 2.
        elif D == 3:
            logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
        return logps

class VESDE(SDE):
    def __init__(self, keys, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
        self.keys = keys

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift_dict = {}
        for key in x.keys():
            drift_dict[key] = torch.zeros_like(x[key])
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift_dict, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape_dict):
        prior_dict = {}
        for key in shape_dict.keys():
            prior_dict[key] = torch.randn(*shape_dict[key]) * self.sigma_max
        return prior_dict

    def prior_logp(self, z):    
        """
        Args:
            z: dict of torch.tensor.
        """
        D = z[self.keys[0]].dim()
        z = torch.cat([z[key] for key in self.keys], D-1)
        shape = z.shape
        N = np.prod(shape[1:])
        if D == 2:
            logps = -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1,)) / (2 * self.sigma_max ** 2)
        elif D == 3:
            logps = -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2)) / (2 * self.sigma_max ** 2)
        return logps

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                    self.discrete_sigmas.to(t.device)[timestep - 1])
        f_dict = {}
        for key in x.keys():
            f_dict[key] = torch.zeros_like(x[key])
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f_dict, G