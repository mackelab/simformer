from typing import Any, Callable, Optional,Sequence
from jax.random import PRNGKey
from jaxtyping import PyTree, Array

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import KeyArray

from abc import abstractmethod
from functools import partial


# This code is used to represent the state of the MCMC chain.
# It is used by the MCMC kernel to keep track of the current state of the chain.
# The state consists of a random number generator key, a value x, and a dictionary of statistics and paramaters.


@register_pytree_node_class
class MCMCState:
    """MCMC state object."""

    def __init__(
        self, key: KeyArray, x: Array, stats: dict = {}, params: dict = {}
    ) -> None:
        """This class represents the state of the MCMC chain.

        Args:
            key (KeyArray): A random number generator key.
            x (Array): Current value of the chain.
            stats (dict, optional): Statistics that are tracked for the chain. Defaults to {}.
            params (dict, optional): Adaptive parameters for the MCMCKernel. Defaults to {}.
        """
        self.key = key
        self.x = x
        self.params = params
        self.stats = stats

    def set_x(self, x: Array):
        self.x = x
        return self

    def set_params(self, params: dict):
        self.params = params
        return self

    def set_stats(self, stats: dict):
        self.stats = stats
        return self

    def set_key(self, key: KeyArray):
        self.key = key
        return self

    def __repr__(self) -> str:
        return f"MCMCState(key={self.key}, x={self.x})"

    # Jax stuff
    def tree_flatten(self):
        return (self.key, self.x, self.stats, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def unzip_vals(states: PyTree[MCMCState] | MCMCState) -> PyTree[Array] | Array:
    """Unzips the states into a tuple of (x, key)"""
    x = jax.tree_map(lambda x: x.x, states, is_leaf=lambda x: isinstance(x, MCMCState))
    return x


class MCMCKernel:
    def __call__(
        self, state: PyTree[MCMCState] | MCMCState
    ) -> PyTree[MCMCState] | MCMCState:
        new_state = jax.tree_map(
            self._update_mcmc_state, state, is_leaf=lambda x: isinstance(x, MCMCState)
        )
        return new_state

    def _update_mcmc_state(self, state: MCMCState) -> MCMCState:
        key1, key2 = jrandom.split(state.key)
        x_new = self._sample(key1, state.x, **state.params)
        stats = self.update_stats(state.x, x_new, state.stats, state.params)
        params = self.update_params(state.x, x_new, state.stats, state.params)
        new_state = MCMCState(key2, x_new, stats, params)
        return new_state

    @abstractmethod
    def _sample(self, key: PRNGKey, x: Array, **params) -> Array:
        """This method should be implemented by the subclass.

        It should return a new sample from the MCMC kernel, given the current state of the chain and additional parameters.

        Args:
            key (PRNGKey): PRNGKey
            x (Array): Old value of the chain.

        Returns:
            Array: New value of the chain.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    # Only required if not symmetric
    def _log_potential(self, x: Array, x_new: Array, **params):
        """Computes the potential of the MCMC kernel, transition from x to x_new.

        Should be the ratio q(x_new|x) / q(x|x_new), if symmetric, this is 1 hence the log potential is 0.

        Args:
            x (Array): Old state of the chain.
            x_new (Array): New state of the chain.
        """
        return 0.0
    
    def init_state(self, key: PRNGKey, init_vals: Array) -> MCMCState:
        """Initialize the state of the MCMC chain"""
        children, tree = jax.tree_util.tree_flatten(init_vals)

        num_keys = len(children)
        keys = jrandom.split(key, num_keys)
        
        params = self.init_params()
        stats = self.init_stats()

        state = jax.tree_map(
            lambda x, k: MCMCState(k, x, params, stats), init_vals, tree.unflatten(keys)
        )
        return state

    def set_potential_fn(self, potential_fn: Callable):
        return self

    def init_params(self) -> dict:
        return {}

    def init_stats(self) -> dict:
        return {}

    def update_params(self, x, x_new, stats, params) -> dict:
        return params

    def update_stats(self, x, x_new, stats, params) -> dict:
        return stats


class PotentialBasedMCMCKernel(MCMCKernel):
    requires_potential: bool = True

    @property
    def potential_fn(self) -> Callable:
        return self._potential_fn

    @potential_fn.setter
    def potential_fn(self, potential_fn: Callable):
        self.set_potential_fn(potential_fn)

    def set_potential_fn(self, potential_fn: Callable):
        self._potential_fn = potential_fn
        return self


class GradientBasedMCMCKernel(PotentialBasedMCMCKernel):
    def set_potential_fn(
        self,
        potential_fn: Callable[..., Any],
        grad_fn: Optional[Callable[..., Any]] = None,
    ):
        def _potential_fn(x):
            return jnp.sum(
                potential_fn(x)
            )  # We choose to average, but could also sum this does not matter

        if grad_fn is None:
            self._potential_value_and_grad_fn = jax.value_and_grad(_potential_fn)
        else:

            def _potential_value_and_grad_fn(x):
                return potential_fn(x), grad_fn(x)

            self._potential_value_and_grad_fn = _potential_value_and_grad_fn

        return super().set_potential_fn(potential_fn)


class MetropolisHastingKernel(PotentialBasedMCMCKernel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self, state: Sequence[MCMCState] | MCMCState
    ) -> Sequence[MCMCState] | MCMCState:
        new_state = super().__call__(state)
        if isinstance(new_state, MCMCState):
            key, key_accept = jax.random.split(new_state.key)
            new_state = new_state.set_key(key)
        else:
            key, key_accept = jax.random.split(new_state[0].key)
            new_state[0] = new_state[0].set_key(key)

        val_old, val_new = unzip_vals((state, new_state))
        # First also check if we cached evaluations in the state!
        logratio = self._mh_hastings_logratio(val_old, val_new)
        accept = jnp.log(jrandom.uniform(key_accept, logratio.shape)) < logratio

        while accept.ndim < val_old[0].ndim:
            accept = jnp.expand_dims(accept, axis=-1)

        # Update the state
        val = jax.tree_map(
            lambda v_new, v_old: jnp.where(accept, v_new, v_old),
            val_new,
            val_old,
        )
        new_state = jax.tree_map(
            lambda s, v: s.set_x(v),
            new_state,
            val,
            is_leaf=lambda x: isinstance(x, MCMCState),
        )

        return new_state

    def _mh_hastings_logratio(
        self,
        val_old: PyTree[Array] | Array,
        val_new: PyTree[Array] | Array,
    ) -> Array:
        logratio = self.potential_fn(*val_new) - self.potential_fn(*val_old)

        # Proposal ratio
        potential_kernel = jax.tree_util.tree_map(
            lambda x_old, x_new: self._log_potential(x_old, x_new),
            val_old,
            val_new,
        )
        logratio += jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            potential_kernel,
        )
        logratio = jnp.nan_to_num(logratio, nan=-jnp.inf)

        return jnp.clip(logratio, a_max=0)


class GaussianKernel(MCMCKernel):
    def __init__(self, step_size: float = 0.1) -> None:
        self.step_size = step_size

    def _sample(self, key, x, step_size=0.1):
        return x + jrandom.normal(key, shape=x.shape) * self.step_size

    def log_potential(self, x, x_new, **params):
        return -0.5 / self.step_size**2 * jnp.sum(
            (x_new - x) ** 2, axis=-1
        ) + jnp.log(self.step_size)


class UniformKernel(MCMCKernel):
    def __init__(self, step_size: float = 0.1) -> None:
        self.step_size = step_size

    def _sample(self, key, x, step_size=0.1):
        return (
            x
            + jrandom.uniform(key, shape=x.shape) * 2 * self.step_size
            - self.step_size
        )


class LangevianDynamicsKernel(GradientBasedMCMCKernel):
    def __init__(self, step_size=0.01) -> None:
        super().__init__()
        self.step_size = step_size
        self._value_grad_cache = (0.0, 0.0)

    def _sample(self, key, x):
        value, grad = self._potential_value_and_grad_fn(x)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=1e8, neginf=-1e8)
        self._value_grad_cache = (value, grad)
        return (
            x
            + self.step_size * grad
            + jrandom.normal(key, shape=x.shape) * jnp.sqrt(2 * self.step_size)
        )


class GaussianMHKernel(MetropolisHastingKernel, GaussianKernel):
    pass


class UniformMHKernel(MetropolisHastingKernel, UniformKernel):
    pass


class LangevianMHKernel(MetropolisHastingKernel, LangevianDynamicsKernel):
    def _log_potential(self, x: Array, x_new: Array, **params):
        _, grad = self._value_grad_cache

        step_size = self.step_size

        potential_x_xnew = -(1 / (4 * step_size)) * jnp.sum(
            (x - x_new - step_size * grad) ** 2, axis=-1
        )
        potential_xnew_x = -(1 / (4 * step_size)) * jnp.sum(
            (x_new - x - step_size * grad) ** 2, axis=-1
        )
        return potential_x_xnew - potential_xnew_x


class HMCKernel(MetropolisHastingKernel, GradientBasedMCMCKernel):
    def __init__(self, step_size=0.1, num_steps=10) -> None:
        super().__init__()
        self.step_size = step_size / num_steps
        self.num_steps = num_steps
        self._cached_energy = (0.0, 0.0)

    def _sample(self, key, x):
        # Sample random momentum
        momentum = jrandom.normal(key, shape=x.shape)
        kinetic_energy = 0.5 * jnp.sum(momentum**2)
        self.x_cache = x
        

        def body_fn(i, carry):
            x, momentum = carry
            _, grad = self._potential_value_and_grad_fn(x)
            momentum += 0.5 * self.step_size * grad
            x += self.step_size * momentum
            _, grad = self._potential_value_and_grad_fn(x)
            momentum += 0.5 * self.step_size * grad
            return (x, momentum)

        (x_new, momentum) = jax.lax.fori_loop(0, self.num_steps, body_fn, (x, momentum))

        new_kinetic_energy = 0.5 * jnp.sum(momentum**2)
        self._cached_energy = (kinetic_energy, new_kinetic_energy)
        return x_new

    def _log_potential(self, x, x_new, **params):
        kinetic_energy, new_kinetic_energy = self._cached_energy
        return kinetic_energy - new_kinetic_energy


class SliceKernel(PotentialBasedMCMCKernel):
    def __init__(self, step_size=0.5, num_steps=100, slice_direction="random") -> None:
        super().__init__()
        self.step_size = step_size
        self.slice_direction = slice_direction
        self.num_steps = num_steps

    def _sample_slice_direction(self, key, x):
        if self.slice_direction == "axis":
            if x.shape[-1] == 1:
                direction = jnp.ones_like(x)
            else:
                axis = jrandom.randint(key, shape=x.shape[-1], minval=0, maxval=x.ndim)
                direction = jnp.zeros_like(x)
                direction = direction.at[..., axis].set(1.0)
        elif self.slice_direction == "random":
            direction = jrandom.normal(key, shape=x.shape)
            direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
        else:
            raise ValueError("Invalid slice direction")
        return direction

    def _sample(self, key, x):
        key_direction, key_bracket, key_shrinkage = jrandom.split(key, 3)
        direction = self._sample_slice_direction(key_direction, x)
        potential = self.potential_fn(x)
        u = jrandom.uniform(key_bracket, shape=potential.shape)
        y = jnp.squeeze(jnp.log(u) + potential)

        # Bracket expansion phase
        def cond_fn(carry):
            i, x, y, direction, mask = carry
            return jnp.any(mask) & (i < self.num_steps)

        def body_fn(carry):
            i, x, y, direction, mask = carry

            # Only evaluate the potential if we need to
            def update_x(x, y, direction):
                x += self.step_size * direction
                potential = self.potential_fn(x)
                mask = potential >= y
                return x, jnp.squeeze(mask)

            # If we don't need to evaluate the potential, just return the current x
            def finished_x(x, y, direction):
                return x, jnp.zeros_like(y, dtype=jnp.bool_)

            if mask.ndim > 0:
                _cond = jax.vmap(jax.lax.cond, in_axes=(None, None, None, 0, 0, 0))
            else:
                _cond = jax.lax.cond
            
            x, mask = _cond(
                mask, update_x, finished_x, x, y, direction
            )

            return (i + 1, x, y, direction, mask)

        _, x_upper, _, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (0, x, y, direction, jnp.ones_like(y, dtype=bool))
        )
        _, x_lower, _, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (0, x, y, -direction, jnp.ones_like(y, dtype=bool))
        )
        # If direction has negative components, then x_lower and x_upper will be swapped

        # Shrinkage phase requires rejection!
        key_shrinkage, key_rejections = jrandom.split(key_shrinkage)
        x_new = (
            jrandom.uniform(key_shrinkage, shape=x.shape) * (x_upper - x_lower)
            + x_lower
        )
        potential_new = self.potential_fn(x_new)
        mask_reject = potential_new <= y


        def cond_fn_reject(carry):
            i,_, _, _, _, _, mask_reject = carry
            return jnp.any(mask_reject) & (i < self.num_steps)

        def body_fn_reject(carry):
            i,key, x_new, x_lower, x_upper, y, mask_reject = carry
            #mask_reject = jnp.expand_dims(mask_reject, axis=-1)
            sign = jnp.sign(direction)
            mask_lower = (x_new * sign <= x * sign) & mask_reject
            mask_upper = (x_new * sign > x * sign) & mask_reject
            x_lower = jnp.where(mask_lower, x_new, x_lower)
            x_upper = jnp.where(mask_upper, x_new, x_upper)

            key, key_reject = jrandom.split(key)

            x_new_reject = (
                jrandom.uniform(key_reject, shape=x_new.shape) * (x_upper - x_lower)
                + x_lower
            )
            x_new = jnp.where(mask_reject, x_new_reject, x_new)
            # We only need to evaluate the potential if the mask is true (i.e. we need to reject)
            # This can be done more efficiently...
            mask_reject = self.potential_fn(x_new) <= y
            # print(self.potential_fn(x_new)[mask_reject], y[mask_reject])
            return (i+1,key, x_new, x_lower, x_upper, y, mask_reject)

        _,_, x_new, _, _, _, _ = jax.lax.while_loop(
            cond_fn_reject,
            body_fn_reject,
            (0, key_rejections, x_new, x_lower, x_upper, y, mask_reject),
        )

        return x_new


class KernelTransformation(MCMCKernel):
    def __init__(self, kernel) -> None:
        self.kernel = kernel

    def __call__(self, state: PyTree | MCMCState) -> PyTree | MCMCState:
        return super().__call__(state)

    def _sample(self, key, x):
        return self.kernel._sample(key, x)

    def _log_potential(self, x: Array, x_new: Array, **params):
        return super()._log_potential(x, x_new, **params)
