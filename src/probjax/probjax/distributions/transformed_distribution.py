import jax.numpy as jnp
import numpy as np

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

import jax
from jax import random
from jax.lax import scan

from .distribution import Distribution
from .constraints import distribution

from typing import Callable, Any, List
from jaxtyping import Array, PyTree

from probjax.core import inverse_and_logabsdet

__all__ = ["TransformedDistribution"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import norm

# TODO: Add support for discrete distributions
# Discrete transformed distributions do not need log_abs_det_jacobian !
# But then we do not need a bijective transformation, just a injective one.
# Bijection do only shuffle the atoms, but do not change the probability mass.



@register_pytree_node_class
class TransformedDistribution(Distribution):
    """
    Creates a transformed distribution by applying an arbitrary callable transformation
    to a base distribution.

    Args:
        base_dist: Base distribution object.
        transformation: Callable transformation that takes samples from the base distribution
            and returns transformed samples.
    """

    arg_constraints = {"base_dist": distribution}

    def __init__(
        self,
        base_dist: Distribution,
        transformation: Callable,
    ):
        self.base_dist = base_dist

        batch_shape = base_dist.batch_shape
        event_shape = base_dist.event_shape

        self.support = base_dist.support
        self._transformation = transformation
        self._inv_and_logdet = inverse_and_logabsdet(transformation)
        
        for _ in range(len(batch_shape)):
            self._transformation = jax.vmap(self._transformation)
            self._inv_and_logdet = jax.vmap(self._inv_and_logdet)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def transform(self, x):
        return self._transformation(x)
    
    def sample(self, key, sample_shape: tuple = ()) -> Array:
        num_samples = max(int(np.prod(sample_shape)), 1)
        samples = self.base_dist.sample(key, (num_samples,))
        if num_samples > 1:
            transform = jax.vmap(self.transform)
        else:
            transform = self.transform
        return transform(samples).reshape(
            sample_shape + self.batch_shape + self.event_shape
        )

    def rsample(self, key, sample_shape=()):
        num_samples = max(int(np.prod(sample_shape)), 1)
        samples = self.base_dist.rsample(key, (num_samples,))
        if num_samples > 1:
            transform = jax.vmap(self.transform)
        else:
            transform = self.transform
        return transform(samples).reshape(
            sample_shape + self.batch_shape + self.event_shape
        )

    def log_prob(self, value):
        shape = value.shape
        value = jnp.asarray(value)
        if value.ndim > 1:
            inv_and_logdet = jax.vmap(self._inv_and_logdet)
        else:
            inv_and_logdet = self._inv_and_logdet

        inv_value, log_det = inv_and_logdet(value)

        inv_value = inv_value.reshape(shape)    
        log_prob = self.base_dist.log_prob(inv_value) + log_det

        if len(self.event_shape) > 0:
            log_prob = log_prob.reshape(shape[: -len(self.event_shape)])
        return log_prob

    def tree_flatten(self):
        return super().tree_flatten()[0], [self._transformation]

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            **dict(zip(cls.arg_constraints.keys(), children)),
            transformation=aux_data[0]
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "base_dist="
            + self.base_dist.__repr__()
            + ", transformation="
            + self._transformation.__repr__()
            + ")"
        )

    def cdf(self, value):
        transformed_value = self.transformation.inv(value)
        return self.base_dist.cdf(transformed_value)

    def icdf(self, value):
        transformed_value = self.base_dist.icdf(value)
        return self.transformation(transformed_value)

    def entropy(self):
        transformed_entropy = self.base_dist.entropy()
        return transformed_entropy - jnp.sum(
            jnp.log(
                jnp.abs(
                    self.transformation.log_abs_det_jacobian(
                        self.transformation.inv(self.base_dist.mean)
                    )
                )
            ),
            axis=-1,
        )
