from typing import Dict, Optional, Any, Tuple


import jax
import jax.numpy as jnp
import jax.random as jrandom

from chex import Numeric, PRNGKey, Array


from probjax.distributions.constraints import Constraint

__all__ = ["Distribution"]

from jax.tree_util import register_pytree_node_class


# Abstract base class for distributions


@register_pytree_node_class
class Distribution:
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    arg_constraints: Dict[str, Constraint] = {}
    support: Constraint = Constraint()
    has_rsample = False
    multivariate = False

    def __init__(
        self,
        batch_shape: tuple = tuple(),
        event_shape: tuple = tuple(),
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

        super().__init__()

    @property
    def batch_shape(self) -> tuple:
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def mean(self) -> Array:
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mean")

    @property
    def median(self) -> Array:
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement median")

    @property
    def mode(self) -> Array:
        """
        Returns the mode of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mode")

    @property
    def variance(self) -> Array:
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement variance")

    @property
    def covariance_matrix(self) -> Array:
        """
        Returns the covariance of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement covariance")

    @property
    def fim(self) -> Array:
        """
        Returns the Fisher information matrix of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement fim")

    @property
    def stddev(self) -> Array:
        """
        Returns the standard deviation of the distribution.
        """
        return jnp.sqrt(self.variance)

    def sample(self, key, sample_shape: tuple = tuple()) -> Array:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        return self.rsample(key, sample_shape)

    def rsample(self, key, sample_shape: tuple = tuple()) -> Array:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError(f"{self.__class__} does not implement rsample")

    def log_prob(self, value: Array) -> Array:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        raise NotImplementedError(f"{self.__class__} does not implement log_prob")

    def prob(self, value: Array) -> Array:
        """
        Returns the probability density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        return jnp.exp(self.log_prob(value))

    def cdf(self, value: Array) -> Array:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        raise NotImplementedError(f"{self.__class__} does not implement cdf")

    def icdf(self, value: Array) -> Array:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        raise NotImplementedError(f"{self.__class__} does not implement icdf")

    def moment(self, n: int) -> Array:
        """
        Returns the nth non-central moment of the distribution, batched over batch_shape.

        Args:
            n (int): order of moment.
        """
        raise NotImplementedError(f"{self.__class__} does not implement moment")

    def entropy(self) -> Array:
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            array of shape batch_shape.
        """
        raise NotImplementedError(f"{self.__class__} does not implement entropy")

    def perplexity(self) -> Array:
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            array of shape batch_shape.
        """
        return jnp.exp(self.entropy())

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                "{}: {}".format(
                    p,
                    self.__dict__[p]
                    if self.__dict__[p].size == 1
                    else self.__dict__[p].size,
                )
                for p in param_names
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"

    # Each distribution will be registered as a PyTree
    def tree_flatten(self):
        return (
            tuple(getattr(self, param) for param in self.arg_constraints.keys()),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(cls.arg_constraints.keys(), children)))
