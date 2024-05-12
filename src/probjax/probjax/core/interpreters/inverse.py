import jax

from jax import lax

from jax.core import Primitive, Jaxpr, JaxprEqn
import functools

from typing import Any, Callable, Optional
import numpy as np
import jax.numpy as jnp
from jax._src.util import safe_map
from jax.custom_derivatives import custom_jvp_call_p
from jax.experimental.pjit import pjit_p

from probjax.core.jaxpr_propagation.utils import ProcessingRule
from probjax.core.jaxpr_propagation.propagate import propagate
from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p


def integer_pow_inverse(x, **params):
    y = params.pop("y")
    return jax.lax.pow_p.bind(x, 1 / y, **params)


def logit(x, **params):
    return jax.lax.log_p.bind(x) - jax.lax.log1p_p.bind(-x)  # type: ignore


_UNIVARITAE_INVERSE_REGISTRY = {
    jax.lax.tanh_p: jax.lax.atanh_p,
    jax.lax.atanh_p: jax.lax.tanh_p,
    jax.lax.sinh_p: jax.lax.asinh_p,
    jax.lax.asinh_p: jax.lax.sinh_p,
    jax.lax.cosh_p: jax.lax.acosh_p,
    jax.lax.acosh_p: jax.lax.cosh_p,
    jax.lax.exp_p: jax.lax.log_p,
    jax.lax.log_p: jax.lax.exp_p,
    jax.lax.sqrt_p: lambda x, **params: jax.lax.pow_p.bind(x, 2.0, **params),
    jax.lax.rsqrt_p: lambda x, **params: 1.0 / jax.lax.pow_p.bind(x, 2.0, **params),
    jax.lax.neg_p: jax.lax.neg_p,
    jax.lax.log1p_p: jax.lax.expm1_p,
    jax.lax.expm1_p: jax.lax.log1p_p,
    jax.lax.erf_p: jax.lax.erf_inv_p,
    jax.lax.erf_inv_p: jax.lax.erf_p,
    jax.lax.conj_p: jax.lax.conj_p,
    jax.lax.real_p: jax.lax.real_p,
    jax.lax.imag_p: jax.lax.imag_p,
    # jax.lax.rev_p: jax.lax.rev_p,
    jax.lax.logistic_p: logit,
    jax.lax.integer_pow_p: integer_pow_inverse,
}


# a * b = c | Lets say that the left and right inverse always gets c as first input and a/b as second!
_BIVARIATE_INVERSE_REGISTRY = {
    jax.lax.mul_p: (
        jax.lax.div_p,
        jax.lax.div_p,
    ),  # a * b = c -> a = c / b -> b = c / a
    jax.lax.div_p: (
        jax.lax.mul_p,
        lambda x, y, **params: jax.lax.div_p.bind(y, x, **params),
    ),  # a / b = c -> a = c * b -> b = a / c
    jax.lax.add_p: (
        jax.lax.sub_p,
        jax.lax.sub_p,
    ),  # a + b = c -> a = c - b -> b = c - a
    jax.lax.sub_p: (
        jax.lax.add_p.bind,
        lambda x, y, **params: jax.lax.sub_p.bind(y, x, **params),
    ),  # a - b = c -> a = c + b -> b = a - c
    jax.lax.pow_p: (
        lambda x, y, **params: jax.lax.pow_p.bind(
            x, 1.0 / y, **params
        ),  # a^b = c -> a = c^(1/b) -> b = log_c(a) = log(a)/log(c)
        lambda x, y, **params: jax.lax.log_p.bind(x) / jax.lax.log_p.bind(y),  # type: ignore
    ),
    jax.lax.pow_p: (
        lambda x, y, **params: jax.lax.pow_p.bind(
            x, 1.0 / y, **params
        ),  # a^b = c -> a = c^(1/b) -> b = log_c(a) = log(a)/log(c)
        lambda x, y, **params: jax.lax.log_p.bind(x) / jax.lax.log_p.bind(y),  # type: ignore
    ),
}

# Custom inverse rules
_CUSTOM_INVERSE_PROCESSING_RULES = {}
_CUSTOM_INVErSE_AND_LOG_DET_RULES = {}


def register_inverse_rule(key):
    def decorator(func):
        nonlocal key
        _CUSTOM_INVERSE_PROCESSING_RULES[key] = func
        return func

    return decorator


def register_inverse_and_log_det_rule(key):
    def decorator(func):
        nonlocal key
        _CUSTOM_INVErSE_AND_LOG_DET_RULES[key] = func
        return func

    return decorator


@register_inverse_rule(jax.lax.concatenate_p)
def invert_concat(eqn, known_invars, known_outvars):
    dim = eqn.params["dimension"]
    out = known_outvars[0]
    in_avals = safe_map(lambda x: x.aval, eqn.invars)
    # out_avals = safe_map(lambda x: x.aval, eqn.outvars)
    split_dimensions = safe_map(lambda x: x.shape[dim], in_avals)
    # Do not trace this with Jax -> Use numpy
    split_indices = np.cumsum(split_dimensions)[:-1].tolist()

    in_vars = jnp.split(
        out,
        split_indices,
        axis=dim,
    )
    return eqn.invars, in_vars


@register_inverse_rule(jax.lax.squeeze_p)
def invert_squeeze(eqn, known_invars, known_outvars):
    in_shape = eqn.invars[0].aval.shape
    out = known_outvars[0]
    return [eqn.invars[0]], [out.reshape(in_shape)]


@register_inverse_rule(jax.lax.broadcast_in_dim_p)
def invert_broadcast_in_dim(eqn, known_invars, known_outvars):
    in_shape = eqn.invars[0].aval.shape
    out = known_outvars[0]
    return [eqn.invars[0]], [out.reshape(in_shape)]


@register_inverse_rule(jax.lax.rev_p)
def invert_rev(eqn, known_invars, known_outvars):
    return eqn.invars, [eqn.primitive.bind(*known_outvars, **eqn.params)]


@register_inverse_rule(jax.lax.gather_p)
def invert_gather(eqn, known_invars, known_outvars):
    # print(known_invars, known_outvars)
    input, index = known_invars
    out = known_outvars[0]

    # print(known_invars, known_outvars)

    if input is None:
        input_aval = eqn.invars[0].aval
        input = jnp.zeros(input_aval.shape, input_aval.dtype)
        # print(input.shape)

    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)

    # TODO CHECH THIS!
    gather_numdim = bind_params["dimension_numbers"]
    scatter_numdim = jax.lax.ScatterDimensionNumbers(
        gather_numdim.offset_dims,
        gather_numdim.collapsed_slice_dims,
        gather_numdim.collapsed_slice_dims,
    )

    # print(subfuns, bind_params)
    # print(eqn.invars[0].aval.shape, input.shape)
    # print(eqn.outvars[0].aval.shape, out.shape)
    # print(index.shape)

    # print(input, index, out, scatter_numdim)
    out = out.reshape(eqn.outvars[0].aval.shape)
    input = jax.lax.scatter(input, index, out, scatter_numdim)

    # input = input.at[index].set(out)

    return [eqn.invars[0]], [input]


@register_inverse_rule(jax.lax.scatter_p)
def invert_scatter(eqn, known_invars, known_outvars):
    index = known_invars[1]
    assert index is not None, "Cannot invert scatter without index!"

    out = known_outvars[0]

    scatter_numdim = eqn.params["dimension_numbers"]
    gather_numdim = jax.lax.GatherDimensionNumbers(
        scatter_numdim.update_window_dims,
        scatter_numdim.inserted_window_dims,
        scatter_numdim.scatter_dims_to_operand_dims,
    )

    slice_sizes = eqn.invars[2].aval.shape
    while len(slice_sizes) < out.ndim:
        slice_sizes = slice_sizes + (1,)
    update_val = jax.lax.gather(out, index, gather_numdim, slice_sizes)

    # print(eqn.params)
    # print(eqn.invars[0].aval.shape, out.shape)
    # print(eqn.invars[2].aval.shape, update_val.shape)

    return [eqn.invars[0], eqn.invars[2]], [out, update_val]


@register_inverse_rule(jax.lax.select_n_p)
def invert_select_n(eqn, known_invars, known_outvars):
    # print(eqn)
    # print(known_invars, known_outvars)
    out = known_outvars[0]
    which = known_invars[:1]
    cases = known_invars[1:]

    in_avals = safe_map(lambda x: x.aval, eqn.invars[1:])

    new_cases = []
    for c, aval in zip(cases, in_avals):
        if c is None:
            new_cases.append(out.astype(aval.dtype))
        else:
            new_cases.append(c)
    # If we do not know which we cannot decide.
    # But we might can reconstruct it!
    if which[0] is None:
        which_var = eqn.invars[0]

    return (
        eqn.invars,
        which + new_cases,
    )


@register_inverse_rule(jax.lax.reshape_p)
def invert_reshape(eqn, _, known_outvars):
    out = known_outvars[0]
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["new_sizes"] = in_aval.shape
    return [eqn.invars[0]], [primitive.bind(out, *subfuns, **bind_params)]


@register_inverse_rule(jax.lax.convert_element_type_p)
def invert_convert_element_type(eqn, known_invars, known_outvars):
    out = known_outvars[0]
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["new_dtype"] = in_aval.dtype
    return [eqn.invars[0]], [primitive.bind(out, *subfuns, **bind_params)]


@register_inverse_rule(jax.lax.slice_p)
def invert_slice(eqn, known_invars, known_outvars):
    input = known_invars[0]
    start_index = eqn.params["start_indices"]
    limit_index = eqn.params["limit_indices"]
    # print(eqn.params)
    invar = eqn.invars[0]
    in_aval = invar.aval
    if input is None:
        input = jnp.zeros(in_aval.shape, in_aval.dtype)
    out1 = known_outvars[0]
    while out1.ndim < input.ndim:
        out1 = jnp.expand_dims(out1, axis=-1)
    # print(input.shape, out1.shape, start_index, limit_index)
    new_input = jax.lax.dynamic_update_slice(input, out1, start_index)
    return [invar], [new_input]


@register_inverse_rule(jax.lax.dynamic_slice_p)
def invert_dynamic_slice(eqn, known_invars, known_outvars):
    input = known_invars[0]
    start_indices = known_invars[1:]

    invar = eqn.invars[0]
    in_aval = invar.aval

    if input is None:
        input = jnp.full(in_aval.shape, jnp.nan, dtype=in_aval.dtype)

    out1 = known_outvars[0]
    # print(out1)
    # while out1.ndim < in_aval.ndim:
    #     out1 = jnp.expand_dims(out1, axis=-1)
    # print(input.shape, out1.shape, start_index, axis, axis)
    new_input = jax.lax.dynamic_update_slice(input, out1, start_indices)

    return [invar], [new_input]


# @register_inverse_rule(jax.lax.dynamic_slice_p)
# def invert_dynamic_slice(eqn, known_invars, known_outvars):
#     invar = eqn.invars[0]
#     slice_sizes = eqn.params["slice_sizes"]
#     out = known_outvars[0]
#     out_shape = eqn.invars[0].aval.shape
#     input = jnp.full(out_shape, jnp.nan)
#     print(input.shape)
#     print(out.shape)


#     new_input = jax.lax.dynamic_update_slice(input, out, known_invars[1])
#     return [invar], [new_input]


def is_univariate(eqn) -> bool:
    return len(eqn.invars) == 1 and len(eqn.outvars) == 1


def is_bivariate(eqn) -> bool:
    return (
        len(eqn.invars) == 2
        and len(eqn.outvars) == 1
        and eqn.primitive in _BIVARIATE_INVERSE_REGISTRY
    )


def has_registered_inverse(eqn, known_invars, known_outvars) -> bool:
    primitive = eqn.primitive

    if primitive is custom_inverse_call_p:
        inv_argnum = eqn.params["inv_argnum"]
        cond1 = known_invars[inv_argnum] is False
        cond2 = all(known_invars[:inv_argnum]) and all(known_invars[inv_argnum + 1 :])
        # print(known_invars, known_outvars, inv_argnum)
        return cond1 and cond2
    else:
        return (
            primitive in _UNIVARITAE_INVERSE_REGISTRY
            or (primitive in _BIVARIATE_INVERSE_REGISTRY and any(known_invars))
            or primitive in _CUSTOM_INVERSE_PROCESSING_RULES
        )


def inverse_cost_fn(eqn, known_invars, known_outvars):
    # Forward computation

    if eqn.primitive is jax.lax.gather_p or eqn.primitive is jax.lax.slice_p:
        # Block gather till necessary!
        return 1.0
    elif eqn.primitive is pjit_p and all(known_outvars):
        # Pjit is a special case
        return 1.5
    elif all(known_invars) and not any(known_outvars):
        return 0
    elif all(known_outvars) and has_registered_inverse(
        eqn, known_invars, known_outvars
    ):
        # If I know all the outputs and the primitive has a registered inverse -> Invert!
        return 0.5
    else:
        return jnp.inf


def value_and_log_det_diagonal(f):
    # This assumes that the jacobian is diagonal!
    # f_sum = lambda *args, **kwargs: jnp.sum(f(*args, **kwargs))
    grad_fn = jax.value_and_grad(f)

    def log_det_fn(*args, **kwargs):
        args_at_least1d = [jnp.atleast_1d(arg) for arg in args]
        args_at_least1d = jnp.broadcast_arrays(*args_at_least1d)
        n_dim = args_at_least1d[0].ndim
        vmaped_grad_fn = grad_fn
        for _ in range(n_dim):
            vmaped_grad_fn = jax.vmap(vmaped_grad_fn)  #
        value, det = vmaped_grad_fn(*args_at_least1d, **kwargs)

        log_det = jnp.log(jnp.abs(det))
        while log_det.ndim > 1:
            log_det = jnp.sum(log_det, axis=-1)
        return value, log_det

    return log_det_fn


def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


def value_and_jacrev(f, x):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


def log_det_multivariate(f):
    # This is expensive!
    def log_det_fn(*args, **kwargs):
        args = [jnp.atleast_1d(arg) for arg in args]
        value, jac = value_and_jacfwd(f, *args)
        sign, log_det = jnp.linalg.slogdet(jac)
        return value, log_det

    return log_det_fn


class InverseProcessingRule(ProcessingRule):
    def __call__(self, eqn, known_invars, known_outvars):
        # print(eqn.primitive, known_invars, known_outvars)

        is_known_invars = safe_map(lambda x: x is not None, known_invars)
        is_known_outvars = safe_map(lambda x: x is not None, known_outvars)

        print(eqn.primitive, is_known_invars, is_known_outvars)

        if eqn.primitive is custom_inverse_call_p and all(is_known_outvars):
            return self._default_custom_inverse_call_apply(
                eqn, known_invars, known_outvars
            )
        elif (
            all(is_known_outvars) and eqn.primitive in _CUSTOM_INVERSE_PROCESSING_RULES
        ):
            return _CUSTOM_INVERSE_PROCESSING_RULES[eqn.primitive](
                eqn, known_invars, known_outvars
            )
        elif (
            not all(is_known_invars)
            and eqn.primitive is jax.experimental.pjit.pjit_p
            #      or eqn.primitive is custom_jvp_call_p
        ):
            return None
        elif all(is_known_invars) and all(is_known_outvars):
            # We already computed all the values -> No need to do anything
            # But we can use these cases to resolve conflicts!
            return self._default_resolve_conflicts(eqn, known_invars, known_outvars)
        elif is_univariate(eqn) and all(is_known_outvars):
            return self._default_univariate_inverse(eqn, known_invars, known_outvars)
        elif is_bivariate(eqn) and all(is_known_outvars) and any(is_known_invars):
            return self._default_bivariate_inverse(eqn, known_invars, known_outvars)
        elif all(is_known_invars):
            return self._default_forward_processing(eqn, known_invars, known_outvars)
        else:
            raise NotImplementedError(f"Cannot invert {eqn}")

    def _default_univariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        if primitive not in _UNIVARITAE_INVERSE_REGISTRY:
            raise NotImplementedError(f"{primitive} is not invertible!")

        inv_primitive = _UNIVARITAE_INVERSE_REGISTRY[primitive]
        if isinstance(inv_primitive, Primitive):
            subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
            invars = inv_primitive.bind(*subfuns, *known_outvars, **bind_params)
        else:
            invars = inv_primitive(*known_outvars, **eqn.params)

        if not isinstance(invars, list):
            invars = [
                invars,
            ]

        return eqn.invars, invars

    def _default_bivariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        input1 = known_outvars[0]
        left_inverse = known_invars[0] is None
        if left_inverse:
            # Left inverse
            input2 = known_invars[1]
        else:
            # Right inverse
            input2 = known_invars[0]

        (left_inverse_fn, right_inverse_fn) = _BIVARIATE_INVERSE_REGISTRY[primitive]
        inv_primitive = left_inverse_fn if left_inverse else right_inverse_fn

        if isinstance(inv_primitive, Primitive):
            subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
            missing_invar = inv_primitive.bind(*subfuns, input1, input2, **bind_params)
        else:
            missing_invar = inv_primitive(input1, input2, **eqn.params)

        if left_inverse:
            return [eqn.invars[0]], [missing_invar]
        else:
            return [eqn.invars[1]], [missing_invar]

    def _default_resolve_conflicts(self, eqn, known_invars, known_outvars):
        # We already computed all the values -> No need to do anything
        # But we can use these cases to resolve conflicts!

        outvars, outvals = self._default_forward_processing(
            eqn, known_invars, known_outvars
        )

        return outvars, outvals

    def _default_forward_processing(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        subfuns, bind_params = primitive.get_bind_params(eqn.params)
        outvals = primitive.bind(*subfuns, *known_invars, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return eqn.outvars, outvals  # type: ignore

    def _default_custom_inverse_call_apply(self, eqn, known_invars, known_outvars):
        inverse_jaxpr = eqn.params["inverse_jaxpr"]
        jaxpr = inverse_jaxpr.jaxpr
        consts = inverse_jaxpr.literals
        inputs = [v if v is not None else known_outvars[0] for v in known_invars]
        out = jax.core.eval_jaxpr(
            jaxpr,
            consts,
            *inputs,
        )
        invars = [
            eqn.invars[i] for i in range(len(eqn.invars)) if known_invars[i] is None
        ]
        inputs = [out[0] for i in range(len(eqn.invars)) if known_invars[i] is None]
        return invars, inputs


class InverseAndLogAbsDetProcessingRule(InverseProcessingRule):
    log_dets = {}

    def __call__(self, eqn, known_invars, known_outvars):
        # print(self.log_dets)
        is_known_invars = safe_map(lambda x: x is not None, known_invars)
        is_known_outvars = safe_map(lambda x: x is not None, known_outvars)

        # print(eqn.primitive, is_known_invars, is_known_outvars)

        if eqn.primitive is custom_inverse_call_p and all(is_known_outvars):
            return self._default_custom_inverse_call_apply(
                eqn, known_invars, known_outvars
            )
        elif (
            all(is_known_outvars) and eqn.primitive in _CUSTOM_INVERSE_PROCESSING_RULES
        ):
            return self._default_custom_rule_apply(eqn, known_invars, known_outvars)
        elif (
            not all(is_known_invars) and eqn.primitive is jax.experimental.pjit.pjit_p
        ):  # or eqn.primitive is custom_jvp_call_p:
            return self._default_pjit(eqn, known_invars, known_outvars)

        elif is_univariate(eqn) and all(is_known_outvars):
            return self._default_univariate_inverse(eqn, known_invars, known_outvars)
        elif is_bivariate(eqn) and all(is_known_outvars) and any(is_known_invars):
            return self._default_bivariate_inverse(eqn, known_invars, known_outvars)
        elif all(is_known_invars):
            return self._default_forward_processing(eqn, known_invars, known_outvars)
        else:
            raise NotImplementedError(f"Cannot invert {eqn}")

    def _default_univariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        if primitive not in _UNIVARITAE_INVERSE_REGISTRY:
            raise NotImplementedError(f"{primitive} is not invertible!")

        inv_primitive = _UNIVARITAE_INVERSE_REGISTRY[primitive]
        if isinstance(inv_primitive, Primitive):

            def f(*args):
                subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
                invars = inv_primitive.bind(*subfuns, *args, **bind_params)

                return jnp.sum(invars)

            eval_fn = value_and_log_det_diagonal(f)
            invars, log_abs_det = eval_fn(*known_outvars)
        else:
            eval_fn = value_and_log_det_diagonal(
                lambda *args: jnp.sum(inv_primitive(*args, **eqn.params))
            )
            invars, log_abs_det = eval_fn(*known_outvars)

        if not isinstance(invars, list):
            invars = [
                invars,
            ]

        log_det_previous = self.log_dets.get(eqn.outvars[0], 0.0)
        self.log_dets[eqn.invars[0]] = log_det_previous + log_abs_det

        return eqn.invars, invars

    def _default_bivariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        input1 = known_outvars[0]
        left_inverse = known_invars[0] is None
        if left_inverse:
            # Left inverse
            input2 = known_invars[1]
        else:
            # Right inverse
            input2 = known_invars[0]

        (left_inverse_fn, right_inverse_fn) = _BIVARIATE_INVERSE_REGISTRY[primitive]

        if left_inverse:
            inv_primitive = left_inverse_fn
        else:
            inv_primitive = right_inverse_fn

        if isinstance(inv_primitive, Primitive):

            def f(*args):
                subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
                invars = inv_primitive.bind(*subfuns, *args, **bind_params)

                return invars

            eval_fn = value_and_log_det_diagonal(f)
            invars, log_abs_det = eval_fn(input1, input2)
        else:
            eval_fn = value_and_log_det_diagonal(
                lambda *args: jnp.sum(inv_primitive(*args, **eqn.params))
            )
            invars, log_abs_det = eval_fn(input1, input2)

        log_det_previous = self.log_dets.get(eqn.outvars[0], 0.0)
        if left_inverse:
            self.log_dets[eqn.invars[0]] = log_det_previous + log_abs_det
            return [eqn.invars[0]], [invars]
        else:
            self.log_dets[eqn.invars[1]] = log_det_previous + log_abs_det
            return [eqn.invars[1]], [invars]

    def _default_pjit(self, eqn, known_invars, outvars):
        if "jaxpr" in eqn.params:
            jaxpr = eqn.params["jaxpr"]
        else:
            jaxpr = eqn.params["call_jaxpr"]

        sub_invars = jaxpr.jaxpr.invars
        sub_outvars = jaxpr.jaxpr.outvars

        subvars = sub_invars + sub_outvars
        vars = eqn.invars + eqn.outvars

        # print(subvars)
        # print(vars)
        for v_sub, v in zip(subvars, vars):
            if v_sub in self.log_dets:
                if not isinstance(v, jax.core.Literal):
                    self.log_dets[v] = self.log_dets[v_sub]

        log_det_previous = sum(
            [
                self.log_dets.get(v, 0.0)
                for v in eqn.outvars
                if not isinstance(v, jax.core.Literal)
            ]
        )

        for v in eqn.invars:
            if not isinstance(v, jax.core.Literal):
                self.log_dets[v] = log_det_previous

        # Pass logdet to outer scope

    def _default_custom_rule_apply(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        if primitive not in _CUSTOM_INVERSE_PROCESSING_RULES:
            raise NotImplementedError(f"{primitive} is not invertible!")

        outvars, outs = _CUSTOM_INVERSE_PROCESSING_RULES[primitive](
            eqn, known_invars, known_outvars
        )
        vars = eqn.invars + eqn.outvars
        log_det_previous = sum([self.log_dets.get(v, 0.0) for v in eqn.outvars])
        for v in outvars:
            self.log_dets[v] = log_det_previous

        return outvars, outs

    def _default_custom_inverse_call_apply(self, eqn, known_invars, known_outvars):
        inverse_jaxpr = eqn.params["inverse_jaxpr"]
        jaxpr = inverse_jaxpr.jaxpr
        consts = inverse_jaxpr.literals
        inputs = [v if v is not None else known_outvars[0] for v in known_invars]
        out = jax.core.eval_jaxpr(
            jaxpr,
            consts,
            *inputs,
        )
        invars = [
            eqn.invars[i] for i in range(len(eqn.invars)) if known_invars[i] is None
        ]
        inputs = [out[0] for i in range(len(eqn.invars)) if known_invars[i] is None]
        log_abs_det = out[-1]
        log_det_previous = sum([self.log_dets.get(v, 0.0) for v in eqn.outvars])
        for v in eqn.invars:
            self.log_dets[v] = log_det_previous + log_abs_det
        out = out[:-1]
        return invars, out
