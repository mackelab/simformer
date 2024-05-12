import jax

import jax.numpy as jnp

jax.numpy.set_printoptions(precision=3, suppress=True)
from jax import core

from jax import linear_util as lu
from functools import partial, update_wrapper

from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, tree_map
from jax.interpreters import ad, batching
from jax._src import ad_util

from jax.core import Primitive, CallPrimitive
from jax._src.util import weakref_lru_cache, cache
from jax._src import util

from typing import Any, Callable
from jax._src.util import safe_map
from jax._src.api_util import (
    flatten_fun_nokwargs,
    argnums_partial,
    flatten_fun_nokwargs,
    shaped_abstractify,
)

from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

# This is a custom primitive that allows us to define custom inverse functions
# While most stuff can be inverted by inverting all primitives for some functions it is necessary or more efficient to define a custom inverse function

custom_inverse_call_p = Primitive("custom_inverse_call_p")
custom_inverse_call_p.multiple_results = True


@custom_inverse_call_p.def_impl
def custom_inverse_call_impl(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        ans = core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.literals, *args)
    return ans


@custom_inverse_call_p.def_abstract_eval
def custom_inverse_call_abstract_eval(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        return forward_jaxpr.out_avals


def custom_inverse_call_lowering(ctx, *args, forward_jaxpr, inverse_jaxpr, **params):
    return mlir.core_call_lowering(
        ctx, *args, name="forward_call", call_jaxpr=forward_jaxpr
    )


mlir.register_lowering(custom_inverse_call_p, custom_inverse_call_lowering)


@jax.util.cache()
def process_jvp(forward_jaxpr, tangents):
    nonzeros = [type(t) is not ad_util.Zero for t in tangents]
    forward_jvp_jaxpr, forward_out_nz = ad.jvp_jaxpr(
        forward_jaxpr, nonzeros, instantiate=False
    )
    nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
    # forward_jvp_jaxpr_ = pe.convert_constvars_jaxpr(forward_jvp_jaxpr.jaxpr)
    return forward_jvp_jaxpr, nonzero_tangents


def custom_inverse_jvp(primals, tangents, forward_jaxpr, inverse_jaxpr, **params):
    forward_jvp_jaxpr, nonzero_tangents = process_jvp(forward_jaxpr, tangents)

    new_primals, new_tangent = core.eval_jaxpr(
        forward_jvp_jaxpr.jaxpr, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )

    return [
        new_primals,
    ], [
        new_tangent,
    ]


def batch_custom_inverse_call(
    spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params
):
    forward_jaxpr = params.pop("forward_jaxpr")
    inverse_jaxpr = params.pop("inverse_jaxpr")

    # We have to batch the jaxprs. For that lets first get the invals and outvals
    in_avals1 = forward_jaxpr.in_avals
    out_avals1 = forward_jaxpr.out_avals

    in_avals2 = inverse_jaxpr.in_avals
    out_avals2 = inverse_jaxpr.out_avals

    # We will batch all the inputs and outputs  (maybe do not batch consts ... )
    in_batched1 = [True] * len(in_avals1)
    out_batched1 = [True] * len(out_avals1)

    in_batched2 = [True] * len(in_avals2)
    out_batched2 = [True] * len(out_avals2)

    # Applies the batching for the jaxprs
    args = [batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]

    # Batched jaxprs
    batched_forward_fn, out_size1 = batching.batch_jaxpr(
        forward_jaxpr,
        axis_size,
        in_batched1,
        out_batched1,
        axis_name,
        spmd_axis_name,
        main_type,
    )
    batched_inverse_fn, _ = batching.batch_jaxpr(
        inverse_jaxpr,
        axis_size,
        in_batched2,
        out_batched2,
        axis_name,
        spmd_axis_name,
        main_type,
    )

    # Update jaxprs with batched ones
    out = custom_inverse_call_p.bind(
        *args,
        forward_jaxpr=batched_forward_fn,
        inverse_jaxpr=batched_inverse_fn,
        **params,
    )

    # Outdim
    out_dims = [0 if b else batching.not_mapped for b in out_size1]

    return out, out_dims


def custom_inverse_transpose(*args, **kwargs):
    return ad.call_transpose(custom_inverse_call_p, *args, **kwargs)


batching.spmd_axis_primitive_batchers[custom_inverse_call_p] = batch_custom_inverse_call
batching.axis_primitive_batchers[custom_inverse_call_p] = partial(
    batch_custom_inverse_call, None
)
ad.primitive_transposes[custom_inverse_call_p] = custom_inverse_transpose
ad.primitive_jvps[custom_inverse_call_p] = custom_inverse_jvp


def is_hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError:
        return False


# TODO: Add support other tracer support!


@jax._src.util.cache()
def trace_forward_inverse(
    f,
    f_inv,
    dyn_args_index,
    inv_argnum,
    in_avals,
    in_tree,
    name,
):
    # print(in_avals)
    # Forward
    f, out_tree = flatten_fun_nokwargs(f, in_tree)  # type: ignore
    debug = pe.debug_info(f.f, in_tree, out_tree, False, name or "<unknown>")
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(f, in_avals, debug)
    forward_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    out_tree = out_tree()

    # Inverse
    f_inv, _ = flatten_fun_nokwargs(f_inv, in_tree)  # type: ignore
    inv_in_avals = list(in_avals)
    i = dyn_args_index.index(inv_argnum)
    inv_in_avals[i] = out_avals[0]
    # print(inv_in_avals)

    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(f_inv, inv_in_avals, debug)
    inverse_jaxpr = core.ClosedJaxpr(jaxpr, consts)

    return forward_jaxpr, inverse_jaxpr, out_tree


class custom_inverse:
    def __init__(self, fun: Callable, inv_argnum=0, static_argnums=None) -> None:
        update_wrapper(self, fun)
        self.fun = fun
        self.static_argnums = static_argnums
        self.inv_argnum = inv_argnum

    def definv(self, inv_fun: Callable) -> Callable:
        def wrapped_inv(*args, **kwargs):
            return inv_fun(*args, **kwargs), jnp.nan

        self.inv_fun = inv_fun
        self.inv_fun_and_log_det = wrapped_inv
        return wrapped_inv

    def definv_and_logdet(self, inv_fun_and_log_det: Callable) -> Callable:
        self.inv_fun_and_log_det = inv_fun_and_log_det
        if not hasattr(self, "inv_fun"):
            self.inv_fun = lambda *args, **kwargs: inv_fun_and_log_det(*args, **kwargs)[
                0
            ]
        return inv_fun_and_log_det

    def inv(self, *args, **kwargs):
        return self.inv_fun(*args, **kwargs)

    def inv_and_logdet(self, *args, **kwargs):
        return self.inv_fun_and_log_det(*args, **kwargs)

    def __call__(self, *args, **params) -> Any:
        name = getattr(self.fun, "__name__", str(self.fun))
        if not self.inv_fun:
            msg = f"No inverse defined for custom_inverse function {name} using definv."
            raise AttributeError(msg)
        inv_name = getattr(self.inv_fun, "__name__", str(self.inv_fun))

        # We can only invert with respect to specific dynamic arguments. All others are assumed to be static!
        f = lu.wrap_init(self.fun, params=params)
        f_inv = lu.wrap_init(self.inv_fun_and_log_det, params=params)

        # Dynamic and static args for forward and inverse
        if self.static_argnums is None:
            dyn_args = args
            dyn_args_index = tuple(i for i in range(len(args)))
        else:
            dyn_args_index = tuple(
                [
                    i
                    for i in range(len(args))
                    if i not in self.static_argnums  # or not is_hashable(args[i])
                ]
            )

            f, dyn_args = argnums_partial(
                f, dyn_args_index, args, require_static_args_hashable=True
            )

            f_inv, _ = argnums_partial(
                f_inv, dyn_args_index, args, require_static_args_hashable=True
            )

        # print(dyn_args, args, self.static_argnums)
        # Flatt stuff for tracing
        args_flat, in_tree = tree_flatten(dyn_args)
        in_avals = tuple(safe_map(shaped_abstractify, args_flat))


        forward_jaxpr, inverse_jaxpr, out_tree = trace_forward_inverse(
            f,
            f_inv,
            dyn_args_index,
            self.inv_argnum,
            in_avals,
            in_tree,
            name,
        )

        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            forward_jaxpr=forward_jaxpr,
            inverse_jaxpr=inverse_jaxpr,
            in_tree=in_tree,
            inv_argnum=dyn_args_index.index(self.inv_argnum),
        )

        return tree_unflatten(out_tree, out_flat)
