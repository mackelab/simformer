import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp

from jax.tree_util import register_pytree_node_class
from jaxtyping import Array
from typing import Union, Sequence

from probjax.distributions.exponential_family import Distribution
from probjax.distributions.independent import Independent
from probjax.distributions.constraints import simplex

@register_pytree_node_class
class Mixture(Distribution):
    arg_constraints = {"mixing_probs": simplex}

    def __init__(self, mixing_probs: Array, component_distributions: Union[Distribution, Sequence[Distribution]]):
        """Initialize a Mixture distribution.

        Args:
            mixing_probs (Array): Mixing probabilities of the components.
            component_distributions (Union[Distribution, Sequence[Distribution]]): Component distributions of the mixture.

        Raises:
            ValueError: If the number of components does not match the number of mixing probabilities.
        """


        component_distributions = Independent(component_distributions, 0)

        num_components = component_distributions.batch_shape[-1]
        batch_shape = component_distributions.batch_shape[:-1]
        event_shape = component_distributions.event_shape

        if mixing_probs.shape[-1] != num_components:
            raise ValueError("Number of components does not match mixing probs")

        mixing_probs = jnp.broadcast_to(mixing_probs, batch_shape + (num_components,))
        self.mixing_probs = mixing_probs
        self.component_distributions = component_distributions
        self.num_components = num_components

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        key_sample, key_cluster_membership = random.split(key, 2)
        shape = sample_shape + self.batch_shape + self.event_shape
        component_samples = self.component_distributions.sample(key_sample, sample_shape)
        cluster_membership = random.categorical(key_cluster_membership, self.mixing_probs, shape= sample_shape + self.batch_shape)
        n_expand = len(self.event_shape) + 1
        cluster_membership = jnp.expand_dims(cluster_membership, axis=tuple(range(-n_expand, 0)))
        samples = jnp.take_along_axis(component_samples, cluster_membership, axis=-len(self.event_shape) - 1)
        return jnp.reshape(samples, shape)

    def rsample(self, key, sample_shape: tuple = ...):
        raise NotImplementedError("Mixture does not support reparameterized sampling, can be done -> implicit reparam.")

    def log_prob(self, value):

        value = jnp.expand_dims(value, axis=-len(self.event_shape) - 1)
        value = jnp.repeat(value, self.num_components, axis=0)
        value = jnp.reshape(value, (-1,) + self.batch_shape + (self.num_components,) + self.event_shape)
        log_component_probs = self.component_distributions.log_prob(value)
        log_mixing_probs = jnp.log(self.mixing_probs)

        log_probs = log_component_probs + log_mixing_probs
        log_probs = logsumexp(log_probs, axis=-1)
        return log_probs

    def cdf(self, value):
        cdf_comp = self.component_distributions.cdf(value)
        return jnp.sum(cdf_comp * self.mixing_probs, axis=-len(self.event_shape) - 1)

    def icdf(self, value):
        icdf_comp = self.component_distributions.icdf(value)
        return jnp.sum(icdf_comp * self.mixing_probs, axis=-len(self.event_shape) - 1)

    @property
    def mean(self):
        mean_comp = self.component_distributions.mean
        return jnp.sum(mean_comp * self.mixing_probs, axis=-len(self.event_shape) - 1)

    @property
    def variance(self):
        variance_comp = self.component_distributions.variance
        t1 = jnp.sum(variance_comp * self.mixing_probs, axis=-len(self.event_shape) - 1)
        t2 = jnp.sum(self.mixing_probs * self.component_distributions.mean ** 2, axis=-len(self.event_shape) - 1)
        t3 = self.mean ** 2
        result = t1 + t2 - t3
        return jnp.where(jnp.isfinite(result), result, jnp.inf)

    # Each distribution will be registered as a PyTree
    def tree_flatten(self):
        flat_components, tree_components = jax.tree_util.tree_flatten(self.component_distributions)
        return (
            (self.mixing_probs,) + tuple(flat_components),
            [tree_components],
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tree_components = aux_data[0]
        return cls(
            children[0], jax.tree_util.tree_unflatten(tree_components, children[1:])
        )

    def __repr__(self) -> str:
        return (
            f"Mixture(mixing_probs={self.mixing_probs.__repr__()}, components={self.component_distributions.__repr__()})"
        )
