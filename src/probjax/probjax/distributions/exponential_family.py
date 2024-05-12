from typing import Dict, Optional, Any, Tuple


import jax
import jax.numpy as jnp
import jax.random as jrandom


from probjax.distributions.constraints import Constraint
from probjax.distributions.distribution import Distribution

from chex import Numeric, PRNGKey, Array

__all__ = ["ExponentialFamily"]

from jax.tree_util import register_pytree_node_class

# Abstract base class for exponential family distributions -> https://en.wikipedia.org/wiki/Exponential_family
# TODO: Add stuff, this currently does nothing.


@register_pytree_node_class
class ExponentialFamily(Distribution):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    @classmethod
    def sufficient_statistic(cls, value: Array) -> Array:
        """
        Computes the sufficient statistics of the distribution.

        Args:
          value: A JAX array representing the value(s) at which to compute the sufficient statistics.

        Returns:
          A JAX array representing the sufficient statistics of the distribution.
        """

        raise NotImplementedError(
            f"{cls.__class__} does not implement sufficient_statistic"
        )

    @classmethod
    def natural_param(cls, params: Array) -> Array:
        """
        Computes the natural parameters of the distribution.

        Args:
          stats: A JAX array representing the sufficient statistics of the distribution.

        Returns:
          A JAX array representing the natural parameters of the distribution.
        """
        raise NotImplementedError(
            "Natural parameters are not implemented for this exponential family distribution."
        )

    @classmethod
    def log_normalizer(cls, params: Array) -> Array:
        """
        Computes the log normalizer of the distribution.

        Args:
          params: A JAX array representing the natural parameters of the distribution.

        Returns:
          A JAX array representing the log normalizer of the distribution.
        """
        raise NotImplementedError(
            "Log normalizer is not implemented for this exponential family distribution."
        )

    @classmethod
    def base_measure(cls, params: Array) -> Array:
        """
        Computes the base measure of the distribution.

        Args:
          params: A JAX array representing the natural parameters of the distribution.

        Returns:
          A JAX array representing the base measure of the distribution.
        """
        raise NotImplementedError(
            "Base measure is not implemented for this exponential family distribution."
        )
