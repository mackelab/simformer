from probjax.core import inverse_and_logabsdet, inverse
from typing import Callable, Sequence

import pytest
import jax
import jax.numpy as jnp


def test_inverse_1d(invertible_function_1d):
    x = jnp.linspace(0, 1, 100)
    y = invertible_function_1d(x)

    mask = jnp.isfinite(y)

    inv_fun = inverse(invertible_function_1d)
    inv_y = inv_fun(y)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(
        x[mask], inv_y[mask], atol=1e-3, rtol=1e-3
    ), "Inverse function value is not correct."


def test_inverse_and_logabsdet_1d(invertible_function_1d):
    x = jnp.linspace(0, 10, 100).reshape(-1, 1)
    y = jax.vmap(invertible_function_1d)(x)
    mask = jnp.isfinite(y).all(-1)

    inv_and_det_fn = inverse_and_logabsdet(invertible_function_1d)
    inv_y, logabsdet = jax.vmap(inv_and_det_fn)(y)

    print(x.shape, inv_y.shape, logabsdet.shape)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(
        x[mask], inv_y[mask], atol=1e-3, rtol=1e-3
    ), "Inverse function value is not correct."

    J = jax.grad(lambda x: invertible_function_1d(x).sum())(x)
    logabsdet_true = -jnp.log(jnp.abs(J)).sum(-1)

    assert jnp.allclose(
        logabsdet[mask], logabsdet_true[mask], atol=1e-3, rtol=1e-3
    ), "Logabsdet is not correct."
