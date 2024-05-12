import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp
from jax.scipy.optimize import minimize

from typing import Callable, Any, List
from jaxtyping import Array, PyTree

from functools import partial

from probjax.distributions import Normal, Distribution


def estimate_ratio_bound(
    log_ratio_fn: Callable, trial_samples: Array, tol=1e-4
) -> Array:
    """
    Estimate the bound for the log density ratio function.

    Args:
        log_ratio_fn: A function that takes in a sample

    Returns:
        The estimated bound.
    """

    @jax.vmap
    def f_min(x):
        return minimize(lambda x: -jnp.sum(log_ratio_fn(x)), x, method="BFGS", tol=tol)

    result = f_min(trial_samples)
    succes = jnp.any(result.success)
    if not succes:
        raise ValueError(
            "Minimization failed, density ratio seems to be unbounded! Please choose a proposal with a larger support i.e. increase variance."
        )
    else:
        return jnp.max(-result.fun).squeeze()


class RejectionSampler:
    def __init__(
        self,
        potential_fn: Callable,
        proposal: Distribution,
        trial_samples_logM: int = 1000,
        batch_size: int = 100,
    ) -> None:
        self._potential_fn = potential_fn
        self._proposal = proposal
        self._log_density_ratio_fn = lambda x: self._potential_fn(
            x
        ) - self._proposal.log_prob(x)
        self._log_M = estimate_ratio_bound(
            self._log_density_ratio_fn,
            proposal.sample(jax.random.PRNGKey(42), (trial_samples_logM,)),
        )
        self._batch_size = batch_size

    # @partial(jax.jit, static_argnums=(0,))
    def run(self, key, num_samples: int = 1, **kwargs) -> Array:
        key, subkey = jax.random.split(key)
        batch_size = self._batch_size
        samples = jnp.empty((num_samples,) + self._proposal.event_shape)

        def cond_fn(state):
            i, key, samples = state
            return i < num_samples

        def body_fn(state):
            i, key, samples = state
            key, key_propose, key_accept = jax.random.split(key, 3)
            proposed_samples = self._proposal.sample(
                key_propose, (batch_size,), **kwargs
            )
            log_ratio = self._log_density_ratio_fn(proposed_samples)
            log_acceptance = log_ratio - self._log_M
            accept = log_acceptance > jax.random.uniform(
                key_accept, shape=log_acceptance.shape
            )
            num_accepted = jnp.sum(accept)
            samples = samples[accept].set(proposed_samples[accept])
            i = i + num_accepted
            return i, key, samples

        _, _, samples = jax.lax.while_loop(cond_fn, body_fn, (0, key, samples))
        return samples
