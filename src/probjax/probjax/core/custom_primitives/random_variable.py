from jax.core import (
    Primitive,
    ClosedJaxpr,
    new_sublevel,
    eval_jaxpr,
    ShapedArray,
)

import jax
import jax.random as jrandom
from jax import tree_util
from jax import linear_util as lu
from jax._src import api_util
from jax._src import ad_util
from jax._src import util
from typing import Hashable, Callable
from jax._src import effects
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters.batching import batch_jaxpr
from jax.interpreters import partial_eval as pe

from jax._src.util import safe_map as map

from functools import partial


from probjax.distributions.distribution import Distribution

__all__ = ["rv", "rv_p"]


def _sample_distribution(dist: Distribution, key, *args, shape=(), **kwargs):
    return dist.sample(key, *args, sample_shape=shape, **kwargs)


def _log_prob_distribution(dist: Distribution, value, *args, **kwargs):
    return dist.log_prob(value=value, *args, **kwargs)


# This maybe should be refactored
@util.cache()
def _sampling_logprobs_jaxprs_with_common_consts(sampling_fn, log_prob_fn):
    wrapped_sampling_fn = lu.wrap_init(sampling_fn)
    in_avals = [
        ShapedArray((2,), jax.numpy.uint32),
    ]  # The PRNG Key!
    in_tree = tree_util.tree_structure(in_avals)
    flat_wrapped_sampling_fn, out_tree = api_util.flatten_fun_nokwargs(  # type: ignore
        wrapped_sampling_fn, in_tree
    )
    debug = pe.debug_info(sampling_fn, in_tree, out_tree, False, "sampling_fn")
    sampling_jaxpr, sampling_out_avals, sampling_consts = pe.trace_to_jaxpr_dynamic(
        flat_wrapped_sampling_fn, in_avals, debug
    )

    wrapped_log_prob_fn = lu.wrap_init(log_prob_fn)
    log_prob_operands = sampling_out_avals
    flat_log_prob_operands, log_prob_in_tree = tree_util.tree_flatten(log_prob_operands)
    flat_wrapped_log_prob_fn, log_prob_out_tree = api_util.flatten_fun_nokwargs(  # type: ignore
        wrapped_log_prob_fn, log_prob_in_tree
    )
    debug = pe.debug_info(
        log_prob_fn, log_prob_in_tree, log_prob_out_tree, False, "log_prob_fn"
    )
    log_prob_jaxpr, log_prob_out_avals, log_prob_consts = pe.trace_to_jaxpr_dynamic(
        flat_wrapped_log_prob_fn, flat_log_prob_operands, debug
    )

    jaxprs = [sampling_jaxpr, log_prob_jaxpr]
    consts = [sampling_consts, log_prob_consts]
    # out_trees = [sampling_out_trees, log_prob_out_trees]

    newvar = jax._src.core.gensym(jaxprs, suffix="_")  # type: ignore
    all_const_avals = [map(api_util.shaped_abstractify, consts) for consts in consts]
    unused_const_vars = [map(newvar, const_avals) for const_avals in all_const_avals]

    def pad_jaxpr_constvars(i, jaxpr):
        prefix = util.concatenate(unused_const_vars[:i])
        suffix = util.concatenate(unused_const_vars[i + 1 :])
        constvars = [*prefix, *jaxpr.constvars, *suffix]
        return jaxpr.replace(constvars=constvars)

    consts = util.concatenate(consts)
    jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
    closed_jaxprs = [
        ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ()) for jaxpr in jaxprs
    ]

    return closed_jaxprs, consts


def rv(dist: Distribution, name: Hashable) -> Callable:
    """This takes a distribution and returns a function that samples from that distribution.


    Args:
        dist (Distribution): Distribution of random variable
        name (Hashable): Name of random variable

    Returns:
        Callable: Sampling function
    """

    def sample_fn(key, *args, **kwargs):
        return _sample_distribution(dist, key, *args, **kwargs)

    def log_prob_fn(value, *args, **kwargs):
        return _log_prob_distribution(dist, value, *args, **kwargs)

    (
        [sampling_fn_jaxpr, log_prob_fn_jaxpr],
        consts,
    ) = _sampling_logprobs_jaxprs_with_common_consts(sample_fn, log_prob_fn)

    def wrapped(*args, **kwargs):
        out = rv_p.bind(
            *consts,
            *args,
            name=name,
            sampling_fn_jaxpr=sampling_fn_jaxpr,
            log_prob_fn_jaxpr=log_prob_fn_jaxpr,
            dist=type(dist),
            intervened=False,
            **kwargs
        )

        return out[0]

    return wrapped


def _rv_impl(*args, **params):
    with new_sublevel():
        call_jaxpr = params["sampling_fn_jaxpr"]
        return eval_jaxpr(call_jaxpr.jaxpr, call_jaxpr.literals, *args)


def _rv_abstract_eval(*args, **params):
    with new_sublevel():
        call_jaxpr = params["sampling_fn_jaxpr"]
        return call_jaxpr.out_avals


# JIT support
def _rv_lowering(ctx, *args, name, sampling_fn_jaxpr, log_prob_fn_jaxpr, **params):
    call_jaxpr = sampling_fn_jaxpr
    return mlir.core_call_lowering(ctx, *args, name=name, call_jaxpr=call_jaxpr)


def _rv_transpose_rule(*args, **kwargs):
    return ad.call_transpose(rv_p, *args, **kwargs)


def _rv_batching_rule(
    spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params
):
    sampling_fn_jaxpr = params.pop("sampling_fn_jaxpr")
    log_prob_fn_jaxpr = params.pop("log_prob_fn_jaxpr")

    # We have to batch the jaxprs. For that lets first get the invals and outvals
    in_avals1 = sampling_fn_jaxpr.in_avals
    out_avals1 = sampling_fn_jaxpr.out_avals

    in_avals2 = log_prob_fn_jaxpr.in_avals
    out_avals2 = log_prob_fn_jaxpr.out_avals

    # We will batch all the inputs and outputs  (maybe do not batch consts ... )
    in_batched1 = [True] * len(in_avals1)
    out_batched1 = [True] * len(out_avals1)

    in_batched2 = [True] * len(in_avals2)
    out_batched2 = [True] * len(out_avals2)

    # Applies the batching for the jaxprs
    args = [batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]

    # Batched jaxprs
    batched_sampling_fn, out_size1 = batch_jaxpr(
        sampling_fn_jaxpr,
        axis_size,
        in_batched1,
        out_batched1,
        axis_name,
        spmd_axis_name,
        main_type,
    )
    batched_log_prob_fn, _ = batch_jaxpr(
        log_prob_fn_jaxpr,
        axis_size,
        in_batched2,
        out_batched2,
        axis_name,
        spmd_axis_name,
        main_type,
    )

    # Update jaxprs with batched ones
    out = rv_p.bind(
        *args,
        sampling_fn_jaxpr=batched_sampling_fn,
        log_prob_fn_jaxpr=batched_log_prob_fn,
        **params
    )

    # Outdim
    out_dims = [0 if b else batching.not_mapped for b in out_size1]

    return out, out_dims


def custom_inverse_jvp(primals, tangents, sampling_fn_jaxpr, **params):
    nonzeros =  [type(t) is not ad_util.Zero for t in tangents]
    forward_jvp_jaxpr, forward_out_nz = ad.jvp_jaxpr(
        sampling_fn_jaxpr, nonzeros, instantiate=False
    )
    nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
    forward_jvp_jaxpr_ = pe.convert_constvars_jaxpr(forward_jvp_jaxpr.jaxpr)

    new_primals, new_tangent = eval_jaxpr(
        forward_jvp_jaxpr_, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )

    return new_primals, new_tangent


rv_p = Primitive("random_variable")
rv_p.multiple_results = True
rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)
batching.spmd_axis_primitive_batchers[rv_p] = _rv_batching_rule
batching.axis_primitive_batchers[rv_p] = partial(_rv_batching_rule, None)
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_transposes[rv_p] = _rv_transpose_rule
ad.primitive_jvps[rv_p] = custom_inverse_jvp
