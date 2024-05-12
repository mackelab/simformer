from typing import Any, Callable, Optional
from jax.random import PRNGKey
from jaxtyping import PyTree, Array

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import KeyArray

from abc import abstractmethod
from functools import partial


# This represents states of particle based sampling algorithms.


@register_pytree_node_class
class State:
    """And state contains just a value and an random variable key, which can be used to change the value."""

    def __init__(
        self,
        key: KeyArray,
        value: Array,
        weights: Optional[Array] = None,
        stats: dict = {},
        params: dict = {},
    ) -> None:
        """This class represents the state of the MCMC chain.

        Args:
            key (KeyArray): A random number generator key.
            value (Array): Current value of the chain.
            stats (dict, optional): Statistics that are tracked for the state. Defaults to {}.
            params (dict, optional): Adaptive parameters for this current state. Defaults to {}.
        """
        self.key = key
        self._value = value
        self._weights = weights
        self.params = params
        self.stats = stats

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, x):
        self._value = x

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, x):
        self._weights = x

    

    def set_x(self, x: Array):
        self.value = x
        return self

    def __repr__(self) -> str:
        return f"MCMCState(key={self.key}, x={self.value})"

    # Jax stuff
    def tree_flatten(self):
        return (self.key, self.value, self.stats, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
