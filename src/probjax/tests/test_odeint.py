from probjax.utils.odeint import get_method_info, get_methods
from probjax.utils.odeint import _odeint as odeint

import pytest

import jax
import jax.numpy as jnp

METHODS = get_methods()

ts_dense = jnp.linspace(0, 5, 1000)

ts = jnp.linspace(0, 5, 100)

def test_odeint_basic_linear_ode(linear_ode_problem, ode_method):
    x0, drift, f_true = linear_ode_problem
    f_approx = odeint(drift, x0, ts_dense, method=ode_method)
    f_true = f_true(ts_dense, x0)
    error = jnp.mean((f_approx - f_true) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"

    
