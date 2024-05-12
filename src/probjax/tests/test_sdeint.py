
import pytest

import jax
import jax.numpy as jnp

from probjax.utils.sdeint import sdeint



# Scalar sde with ground truth solution

t = jnp.linspace(0., 10., 200)


def test_sdeint_scalar(sde_method, scalar_sde_problem):
    x0, f, g, f_true = scalar_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, W = sdeint(key,f,g, x0, t, method=sde_method, return_brownian=True)
    f_sol = f_true(W,t,x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"

def test_sdeint_2d(sde_method, two_dimensional_sde_problem):
    x0, f, g, f_true = two_dimensional_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, W = sdeint(key,f,g, x0, t, method=sde_method, return_brownian=True)
    f_sol = f_true(W,t,x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"









