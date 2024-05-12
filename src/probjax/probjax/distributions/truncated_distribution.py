import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution
from .constraints import real, positive, unit_interval, interval, distribution

__all__ = ["TruncatedNormal"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import truncnorm


@register_pytree_node_class
class TruncatedDistribution(Distribution):
    
    arg_constraints = {"base_dist": distribution, "a": real, "b": real}

    def __init__(self, base_dist, a, b):
        self.base_dist = base_dist

        self.a = jnp.minimum(a, b)
        self.b = jnp.maximum(a, b)

        batch_shape = base_dist.batch_shape
        event_shape = base_dist.event_shape

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        cdf_left = self.base_dist.cdf(self.a)
        cdf_right = self.base_dist.cdf(self.b)
        average_acceptance_probability = (
            (cdf_right - cdf_left).prod(-len(self.event_shape)).mean()
        )
        assert (
            average_acceptance_probability < 1e-4
        ), "Average acceptance probability is too low"
        num_samples = (sample_shape + self.batch_shape).prod()
        expected_required_samples = num_samples / average_acceptance_probability

        total_samples = []
        while len(total_samples) < num_samples:
            key, subkey = random.split(key)
            samples = self.base_dist.sample(subkey, (expected_required_samples,))
            accepted_samples = jnp.where((samples >= self.a) & (samples <= self.b))
            total_samples.append(samples[accepted_samples])
        total_samples = jnp.concatenate(total_samples, axis=0)
        return total_samples[:num_samples].reshape(shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value) - jnp.log(
            self.cdf(self.b) - self.cdf(self.a)
        )
