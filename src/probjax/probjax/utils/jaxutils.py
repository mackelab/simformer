from typing import Callable, Union, Tuple
from jaxtyping import PyTree, Array

import jax
import jax.numpy as jnp
import numpy as np
from functools import wraps, partial

from jax._src.flatten_util import ravel_pytree
from jax._src.api_util import flatten_fun_nokwargs as flatten_fun_nokwargs_

from jax import linear_util as lu


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat
    
@lu.transformation
def ravel_arg_(unravel, index, *args):
    flat_arg_i = args[index]
    arg_i = unravel(flat_arg_i)
    args = args[:index] + (arg_i,) + args[index+1:]
    ans = yield args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat      


@lu.transformation
def ravel_args_(unravel, args_flat):
    args = unravel(args_flat)
    ans = yield args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


@lu.transformation_with_aux
def flatten_args_(in_tree, *flat_args):
    args = jax.tree_util.tree_unflatten(in_tree, flat_args)
    ans = yield (args,), {}
    ans_flat = jax.tree_util.tree_flatten(ans)
    yield ans_flat


def flatten_fun(fun: Callable, in_tree: PyTree) -> Callable:
    """Flattens the input arguments of a function. Meaning than all abstract inputs are flattened into a list of arrays.

    Args:
        fun (Callable): Function to be flattened
        in_tree (PyTree): In tree of the functions input arguments

    Returns:
        Tuple[Callable]: The flattened function
    """

    def fun_new(*args):
        f_flat, out_tree = flatten_args_(lu.wrap_init(fun), in_tree)
        out = f_flat.call_wrapped(*args)
        return jax.tree_util.tree_unflatten(out_tree(), out)

    return fun_new


def ravel_args(in_vals: PyTree) -> Tuple[Array, Callable]:
    """_summary_

    Args:
        in_vals (PyTree): _description_

    Returns:
        Tuple[Array, Callable]: _description_
    """
    flat_vals, unflatten = ravel_pytree(in_vals)
    return flat_vals, unflatten


def ravel_fun(fun: Callable, unravel) -> Callable:
    return ravel_args_(lu.wrap_init(fun), unravel).call_wrapped

def ravel_arg_fun(fun: Callable, unravel, index: int) -> Callable:
    return ravel_arg_(lu.wrap_init(fun), unravel, index).call_wrapped


def ravel_first_arg_fun(fun: Callable, unravel) -> Callable:
    return ravel_first_arg_(lu.wrap_init(fun), unravel).call_wrapped


# def sliced_potential_fn(flatten_potential_fn, loc, direction):
#     """Returns a function that slices the potential function in a given direction."""

#     def _sliced_potential_fn(t):
#         return flatten_potential_fn(loc + t * direction)

#     return _sliced_potential_fn


# def conditional_potential_fn(flatten_potential_fn, x, indices):
#     """Returns a function that slices the potential function in a given direction."""

#     def _conditional_potential_fn(sub_x):
#         return flatten_potential_fn(x.at[indices].set(sub_x))

#     return _conditional_potential_fn
