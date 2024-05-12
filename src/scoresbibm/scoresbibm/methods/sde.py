import jax
import jax.numpy as jnp

from probjax.distributions import Independent
from probjax.distributions.discrete import Empirical
from probjax.distributions.sde import VESDE, VPSDE


def init_sde_related(data, name="vpsde", **kwargs):
    """ Initialize the sde and related functions."""
    # VPSDE 
    if name.lower() == "vpsde":
        p0 = Independent(Empirical(data), 1)
        beta_max = kwargs.get("beta_max",10.)
        beta_min = kwargs.get("beta_min", 0.01)
        sde = VPSDE(p0, beta_max=beta_max, beta_min=beta_min)
        T_max = kwargs.get("T_max", 1.)
        T_min = kwargs.get("T_min", 1e-5)
        scale_min = kwargs.get("scale_min", 0)

        # Train weight function
        def weight_fn(t):
            t = t.reshape(-1, 1)
            return jnp.clip(1-jnp.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t) ,a_min = 1e-4)

        # Model output scale function
        def output_scale_fn(t, x):
            scale = jnp.clip(jnp.sqrt(jnp.sum(sde.marginal_variance(t[..., None], x0=jnp.ones_like(x)), -1)), scale_min)
            return 1/scale[..., None] * x
    elif name.lower() == "vesde":
        p0 = Independent(Empirical(data), 1)
        sigma_min = kwargs.get("sigma_min", 0.01)
        sigma_max = kwargs.get("sigma_max", 10.)
        sde = VESDE(p0, sigma_min=sigma_min, sigma_max=sigma_max)
        T_max = kwargs.get("T_max", 1.)
        T_min = kwargs.get("T_min", 1e-5)
        scale_min = kwargs.get("scale_min", 1e-3)

        # Train weight function
        def weight_fn(t):
            t = t.reshape(-1, 1)
            return sde.diffusion(t, jnp.ones((1,)))**2

        # Model output scale function
        def output_scale_fn(t, x):
            scale = jnp.clip(jnp.sqrt(jnp.sum(sde.marginal_variance(t[..., None], x0=jnp.ones_like(x)), -1)), scale_min)
            return 1/scale[..., None] * x
    else:
        raise NotImplementedError()

    return sde, T_min, T_max, weight_fn, output_scale_fn