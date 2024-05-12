from probjax.distributions import Distribution, Normal, Independent
from probjax.nn.coupling import CouplingMLP
from probjax.nn.bijective import rational_quadratic_spline
from probjax.nn.helpers import Rotate, Flip, SinusoidalEmbedding
from probjax.utils.odeint import odeint

import jax
import jax.numpy as jnp

from functools import partial

import haiku as hk

import pytest


input_dim = 2
p = Independent(Normal(jnp.zeros(input_dim), jnp.ones(input_dim)), 1)

# Mixture of Gaussians target distribution


def target_samples(input_dim):
    p = Independent(Normal(jnp.zeros(input_dim), jnp.ones(input_dim)), 1)
    target_samples1 = p.sample(jax.random.PRNGKey(0), (500,)) + 5.0
    target_samples2 = p.sample(jax.random.PRNGKey(1), (500,)) - 5.0
    xs = jnp.concatenate([target_samples1, target_samples2], axis=0)
    return xs


def base_dist(input_dim):
    p = Independent(Normal(jnp.zeros(input_dim), jnp.ones(input_dim)), 1)
    return p


def affine_bijector(loc_and_scale, x):
    loc, scale = jnp.split(loc_and_scale, 2, axis=-1)
    scale = jnp.exp(scale) + 0.5

    return loc + scale * x


def spline_bijector(params, x):
    return rational_quadratic_spline(
        params,
        x,
        range_min_x=-4,
        range_max_x=4,
        range_min_y=-15.0,
        range_max_y=15.0,
    )


def affine_coupling_transform(input_dim):
    bijector_dim = 2 * (input_dim // 2)
    split_dim = input_dim // 2

    @hk.without_apply_rng
    @hk.transform
    def forward(x):
        nn = hk.Sequential(
            [
                CouplingMLP(split_dim, affine_bijector, bijector_dim),
                Flip(),
                CouplingMLP(split_dim, affine_bijector, bijector_dim),
                Flip(),
                CouplingMLP(split_dim, affine_bijector, bijector_dim),
                Flip(),
                CouplingMLP(split_dim, affine_bijector, bijector_dim),
                Flip(),
                CouplingMLP(split_dim, affine_bijector, bijector_dim),
            ]
        )
        return nn(x)

    params = forward.init(jax.random.PRNGKey(0), jnp.zeros((input_dim)))
    T = forward.apply

    return params, T


def spline_coupling_transform(input_dim):
    num_bins = 4
    bijector_dim = 3 * num_bins
    split_dim = input_dim // 2

    spline = jax.vmap(spline_bijector, in_axes=(None, 0))

    @hk.without_apply_rng
    @hk.transform
    def forward(x):
        nn = hk.Sequential(
            [
                CouplingMLP(split_dim, spline, bijector_dim),
                Flip(),
                CouplingMLP(split_dim, spline, bijector_dim),
                Flip(),
            ]
        )
        return nn(x)

    init, apply = forward.init, forward.apply
    params = init(jax.random.PRNGKey(0), jnp.ones((input_dim,)))
    T = apply

    return params, T


def spline_gaussianization_transform(input_dim):
    spline_based = jax.vmap(spline_bijector, in_axes=(-1, -1))

    @hk.without_apply_rng
    @hk.transform
    def f(x):
        r1 = Rotate(jax.random.PRNGKey(0), input_dim)
        r2 = Rotate(jax.random.PRNGKey(1), input_dim)
        r3 = Rotate(jax.random.PRNGKey(2), input_dim)

        params1 = hk.get_parameter(
            "params_spline1",
            shape=(12, input_dim),
            init=hk.initializers.RandomNormal(0.1),
        )
        params2 = hk.get_parameter(
            "params_spline2",
            shape=(12, input_dim),
            init=hk.initializers.RandomNormal(0.1),
        )
        params3 = hk.get_parameter(
            "params_spline3",
            shape=(12, input_dim),
            init=hk.initializers.RandomNormal(0.1),
        )
        params4 = hk.get_parameter(
            "params_spline4",
            shape=(12, input_dim),
            init=hk.initializers.RandomNormal(0.1),
        )

        x1 = spline_based(params1, x)
        x2 = r1(x1)
        x3 = spline_based(params2, x2)
        x4 = r2(x3)
        x5 = spline_based(params3, x4)
        x6 = r3(x5)
        x7 = spline_based(params4, x6)

        return x7

    init, apply = f.init, f.apply

    params = init(jax.random.PRNGKey(0), jnp.ones((input_dim,)))

    T = apply

    return params, T


def continous_transform(input_dim):
    def net(t, x):
        # print("Net", t,x)
        t = jnp.array(t).reshape(-1)
        t = SinusoidalEmbedding(8)(t)
        return hk.nets.MLP([50, 50, input_dim], activation=jnp.tanh)(
            jnp.concatenate([t, x], axis=-1)
        )

    init, f = hk.without_apply_rng(hk.transform(net))
    params = init(jax.random.PRNGKey(0), jnp.ones(1), jnp.ones((input_dim,)))

    def T(params, x):
        ys = odeint(lambda t, x: f(params, t, x), x, jnp.linspace(0, 1, 100))
        return ys[-1]

    return params, T


import optax
from optax import adam
from probjax.distributions.transformed_distribution import TransformedDistribution


def fit(params, p, T, xs):
    optimizer = adam(1e-3)
    opt_state = optimizer.init(params)

    # Train
    def loss_fn(params, x):
        q = TransformedDistribution(p, lambda x: T(params, x))
        return -q.log_prob(x).mean()

    @jax.jit
    def step(params, opt_state, x):
        loss, g = jax.value_and_grad(loss_fn)(params, x)
        updates, updated_state = optimizer.update(g, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, updated_state, loss

    def train(params, opt_state, xs):
        for i in range(500):
            params, opt_state, loss = step(params, opt_state, xs)
        return params

    params = train(params, opt_state, xs)

    return TransformedDistribution(p, lambda x: T(params, x))
