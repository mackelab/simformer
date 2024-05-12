import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.util import safe_map as map
from .marcov_kernels import MCMCKernel, MCMCState, unzip_vals

from typing import Any, Callable, Tuple, Union, Sequence
from jaxtyping import PyTree, Array

from functools import partial
from itertools import accumulate

from probjax.utils.jaxutils import flatten_fun, ravel_fun


class MCMC:
    def __init__(
        self,
        kernel: PyTree[MCMCKernel] | MCMCKernel,
        potential_fn: Callable[[PyTree[Array] | Array], Array],
    ) -> None:
        self.kernel = kernel
        self.potential_fn = potential_fn

    def _check_potential_fn(self, x: PyTree[Array] | Array) -> Array:
        try:
            self.potential_fn(x)
        except:
            assert (
                False
            ), "Potential function must evaluatable given init_vals as input."

    def _check_kernel_tree(self, in_tree) -> None:
        # Should have the same PyTree structure
        flat_kernel, kernel_tree = jax.tree_flatten(self.kernel)
        assert (
            kernel_tree.num_leaves == 1 or in_tree == kernel_tree
        ), "The kernel must only have a single leave or the same PyTree structure as init_vals!"

        if kernel_tree.num_leaves == 1:
            return flat_kernel[0]
        else:
            return flat_kernel


    @partial(jax.jit, static_argnums=(0,))
    def run(self, state: PyTree[MCMCState] | MCMCState, num_steps: int):
        # Flat the state
        flat_state, in_tree = jax.tree_flatten(
            state, is_leaf=lambda x: isinstance(x, MCMCState)
        )
        # MCMC kernel flatten and check compatibility
        flat_kernel = self._check_kernel_tree(in_tree)

        # Flatten the potential function, give it to all kernels (that might need it)
        flatten_potential_fn = flatten_fun(self.potential_fn, in_tree)
        flat_kernel = jax.tree_map(
            lambda x: x.set_potential_fn(flatten_potential_fn), flat_kernel
        )

        def body_fn(i, carry):
            state = carry
            new_state = jax.tree_map(lambda kernel, x: kernel(x), flat_kernel, state)
            return new_state

        out_state = jax.lax.fori_loop(0, num_steps, body_fn, flat_state)
        out_state = jax.tree_util.tree_unflatten(in_tree, out_state)
        vals = unzip_vals(out_state)
        return vals, out_state
