import pytest

import jax
import jax.numpy as jnp
from jax import random

import haiku as hk

key = random.PRNGKey(0)


# Invertile function testcase fixtures


# Simple invertible 1d transformations
def for_loop_sum(x):
    x0 = x
    for i in range(10):
        x0 += 2
    return x0


def for_loop_mul(x):
    x0 = x
    for i in range(10):
        x0 *= 2
    return x0


# Reshape and revert
def reshape_and_revert(x):
    y = x.reshape((1, 1, 1, 1, 1) + x.shape)
    return y.reshape(x.shape)


def broad_cast_and_revert(x):
    y = x[..., None, None, None, None]
    return y[..., 0, 0, 0, 0]


# jnp.where does not yet work! -> thus also not leaky relu and so on...
INVERTIBLE_FUNCTIONS_1d = [
    jnp.log,
    jnp.log2,
    jnp.log10,
    jnp.log1p,
    # lambda x: jnp.logaddexp(x,1.), # TODO jnp.where does not yet work!
    # lambda x: jnp.logaddexp2(x,1.), # TODO jnp.where does not yet work!
    jnp.exp,
    jnp.exp2,
    # jnp.flip, # TODO ERROR
    for_loop_sum,
    for_loop_mul,
    reshape_and_revert,
    broad_cast_and_revert,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x**3,
    lambda x: x * 2,
    lambda x: x / 2,
]


@pytest.fixture(params=INVERTIBLE_FUNCTIONS_1d)
def invertible_function_1d(request):
    return request.param


# Flows fixtures ---------------------------------------------------------
from config_flows import (
    target_samples,
    base_dist,
    affine_coupling_transform,
    spline_coupling_transform,
    spline_gaussianization_transform,
    continous_transform,
)


@pytest.fixture(
    params=[
        affine_coupling_transform,
        spline_coupling_transform,
        spline_gaussianization_transform,
        continous_transform,
    ]
)
def transform(request):
    return request.param


@pytest.fixture()
def target_xs():
    return target_samples


@pytest.fixture()
def base_distribution():
    return base_dist


@pytest.fixture(params=[2, 8])
def input_dim(request):
    return request.param


# SDE problems fixtures ---------------------------------------------------------
from probjax.utils.sdeint import get_methods

METHODS = get_methods()


@pytest.fixture(params=METHODS, ids=METHODS)
def sde_method(request):
    return request.param


@pytest.fixture
def scalar_sde_problem():
    x0 = jnp.array([0.5])

    def f(t, x):
        return -((1 / 10) ** 2) * jnp.sin(x) * jnp.cos(x) ** 3

    def g(t, x):
        return 1 / 10 * jnp.cos(x) ** 2

    def f_true(Wt, t, x0):
        return jnp.arctan(1 / 10 * Wt + jnp.tan(x0))

    return x0, f, g, f_true


@pytest.fixture
def two_dimensional_sde_problem():
    x0 = jnp.array([0.5, 0.5])

    def f2(t, x):
        return 0.5 * x * (1 - x) * (1 - 2 * x)

    def g2(t, x):
        return x * (1 - x)

    def f2_true(Wt, t, x0):
        return 1 / (1 + jnp.exp(-Wt + jnp.log(x0 / (1 - x0)).reshape(-1, 2)))

    return x0, f2, g2, f2_true


# ODE problems fixtures ---------------------------------------------------------

from probjax.utils.odeint import get_methods

METHODS = get_methods()


A1 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # Peridoic
A2 = jnp.array([[0.0, 1.0], [-1.0, -1.0]])  # Stable
A3 = jnp.array([[0.0, 1.0], [-1.0, 1.0]])  # Unstable
A4 = jax.random.normal(key, (5, 5)) * 0.1  # Random


@pytest.fixture(params=METHODS, ids=METHODS)
def ode_method(request):
    return request.param


@pytest.fixture
def ode_methods():
    return METHODS


@pytest.fixture(
    params=[A1, A2, A3, A4],
    ids=["linear_periodic_2d", "linear_stable_2d", "linear_unstable_2d", "random_5d"],
)
def linear_ode_problem(request):
    A = request.param
    x0 = jnp.ones((A.shape[0],))

    def f(t, x):
        return jnp.dot(A, x)

    def true_f(t, x0):
        E = A.reshape((1,) + A.shape)
        t = t.reshape(t.shape + (1,) * len(A.shape))
        Phi = jax.scipy.linalg.expm(E * t)
        return jnp.dot(Phi, x0)

    return x0, f, true_f
