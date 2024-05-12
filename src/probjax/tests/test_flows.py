import pytest

import jax
import jax.numpy as jnp

from probjax.core import inverse, inverse_and_logabsdet


def test_transform_inv_and_lodet(input_dim, transform):

    x = jnp.ones((input_dim,))
    params, T = transform(input_dim)
    f = lambda x: T(params, x)
    f_inv = inverse(f)
    f_inv_lodet = inverse_and_logabsdet(f)

    y = f(x)
    x_recon = f_inv(y)
    x_recon2, logabsdet = f_inv_lodet(y)

    assert jnp.allclose(
        x,
        x_recon,
        atol=1e-2,
        rtol=1e-2,
    ), "inverse does not invert correctly"
    assert jnp.allclose(
        x,
        x_recon2,
        atol=1e-2,
        rtol=1e-2,
    ), "inverse_and_logabsdet does not invert correctly"

    jac = jax.jacfwd(f)(x).reshape(input_dim, input_dim)
    log_abs_det_jac = jnp.linalg.slogdet(jac)[1]

    assert jnp.allclose(
        -log_abs_det_jac,
        logabsdet,
        atol=1e-2,
        rtol=1e-2,
    ), "inverse_and_logabsdet does not compute correct logabsdet"
