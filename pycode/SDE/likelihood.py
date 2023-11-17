import functools
import torch
import numpy as np

from scipy import integrate
from pycode.SDE import utils



def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(data, t, eps):
        sorted_key = list(data.keys())
        sorted_key.sort()

        with torch.enable_grad():
            for key in sorted_key:
                data[key] = data[key].requires_grad_(True)
            
            output = fn(data, t)
            
            fn_eps = 0
            for key in sorted_key:
                fn_eps += torch.sum(output[key] * eps[key])
            
            grad_fn_eps = torch.autograd.grad(fn_eps, [data[key] for key in sorted_key])
        
        for key in sorted_key:
            data[key] = data[key].requires_grad_(False)
        
        for i, key in enumerate(sorted_key):
            if i == 0:
                div = torch.sum(grad_fn_eps[i] * eps[key], dim=tuple(range(1, len(data[key].shape))))
            else:
                div += torch.sum(grad_fn_eps[i] * eps[key], dim=tuple(range(1, len(data[key].shape))))
        return div

    return div_fn


def get_likelihood_fn(sde, inverse_scaler, shape_dict, hutchinson_type='Rademacher',
                        rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        inverse_scaler: The inverse data normalizer.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
        eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
        A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = utils.get_score_fn(sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            model: A score model.
            data: A PyTorch tensor.

        Returns:
            bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
            z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
                probability flow ODE.
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """

        key_list = list(shape_dict.keys())
        key_list.sort()
        B = data[key_list[0]].shape[0]

        for key in key_list:
            shape_dict[key] = data[key].shape

        device = data[key_list[0]].device

        from_flattened_numpy = functools.partial(utils.from_flattened_numpy, key_list=key_list, shape_dict=shape_dict)
        to_flattened_numpy = functools.partial(utils.to_flattened_numpy, key_list=key_list)
        
        with torch.no_grad():
            epsilon_dict = {}
            for key in data.keys():
                shape = data[key].shape
                if hutchinson_type == 'Gaussian':
                    epsilon_dict[key] = torch.randn_like(data[key])
                elif hutchinson_type == 'Rademacher':
                    epsilon_dict[key] = torch.randint_like(data[key], low=0, high=2).float() * 2 - 1.
                else:
                    raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = from_flattened_numpy(x[:-B])
                for key in sample.keys():
                    sample[key] = sample[key].to(device).type(torch.float32)
                vec_t = torch.ones(B, device=device) * t
                drift = to_flattened_numpy(drift_fn(model, sample, vec_t))
                logp_grad = div_fn(model, sample, vec_t, epsilon_dict).detach().cpu().numpy().reshape((-1,))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(data), np.zeros((B,))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-B])
            for key in z.keys():
                z[key] = z[key].to(device).type(torch.float32)
                
            delta_logp = torch.from_numpy(zp[-B:].reshape((B,))).to(device).type(torch.float32)
            prior_logp = sde.prior_logp(z)
            logp = prior_logp + delta_logp
            # bpd = -logp / np.log(2)
            # N = np.prod(shape[1:])
            # bpd = bpd / N
            # # A hack to convert log-likelihoods to bits/dim
            # offset = 7. - inverse_scaler(-1.)
            # bpd = bpd + offset
            return logp, z, nfe

    return likelihood_fn