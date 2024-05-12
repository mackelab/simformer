import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.util import safe_map as map
from .marcov_kernels import MCMCKernel, MCMCState, GaussianKernel

from typing import Any, Callable, Tuple, Union, Sequence
from jaxtyping import PyTree, Array

from functools import partial
from itertools import accumulate


def geometric_tempering(initial_potential_fn, final_potential_fn, num_steps):
    def potential_fn(t, x):
        alpha = t / (num_steps - 1)
        return (1 - alpha) * initial_potential_fn(x) + alpha * final_potential_fn(x)

    return potential_fn

def no_tempering(initial_potential_fn, final_potential_fn, num_steps):
    def potential_fn(t, x):
        return final_potential_fn(x)

    return potential_fn

def get_tempering_fn(tempering):
    if tempering == "geo":
        return geometric_tempering
    elif tempering == "none":
        return no_tempering
    else:
        raise ValueError("tempering must be one of 'geometric' or 'no'")


# class SMCState:
#     def __init__(self, key, particles, weights) -> None:
#         self.key = key
#         self.particles = particles
#         self.weights = weights


class SMC:
    def __init__(
        self, initial_distribution, potential_fn, marcov_kernel, tempering="geo",
    ) -> None:
        self.initial_distribution = initial_distribution
        self.potential_fn = potential_fn
        self.tempering_fn = get_tempering_fn(tempering)
        self.marcov_kernel = marcov_kernel

    @partial(jax.jit, static_argnums=(0,))
    def run(self, key, num_steps, num_particles):
        init_key, iter_key = jrandom.split(key)
        init_particles = self.initial_distribution.sample(init_key, (num_particles,))
        init_weights = jnp.ones((num_particles,)) / num_particles

        target_potential_fn = self.path_fn(
            lambda x: self.initial_distribution.log_prob(x),
            self.potential_fn,
            num_steps,
        )

        def body_fn(i, state):
            key, particles, weights = state
            key, key_resample, key_propose = jrandom.split(key)
            re_sample_index = jrandom.categorical(
                key_resample, weights, shape=(num_particles,)
            )
            particles = particles[
                re_sample_index
            ]  # Note this also makes the weights uniform, so we can ignore them
            new_particles = self.marcov_kernel.sample(key_propose, particles)
            weights = (
                target_potential_fn(i, new_particles)
                - target_potential_fn(i - 1, new_particles)
                # + self.marcov_kernel.log_potential(new_particles, particles)
                # - self.marcov_kernel.log_potential(particles, new_particles)
            )
            weights = jax.nn.softmax(weights)
            return key, new_particles, weights

        _, particles, weights = jax.lax.fori_loop(
            1, num_steps, body_fn, (iter_key, init_particles, init_weights)
        )
        return particles, weights
