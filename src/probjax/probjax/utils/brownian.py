import jax
import jax.numpy as jnp
from jax import lax
from jax import core
import jax.random as jrandom

from jaxtyping import Array, Float, PyTree
from jax.random import PRNGKey

from functools import partial

# Iterated integrals


@jax.jit
def iterated_ito_integral_general(key: PRNGKey, dW: Array, dt: Array, n: int = 5):
    """Matrix I approximating repeated Ito integrals based on the method of Kloeden, Platen and Wright (1992).

    Args:
        key (PRNGKey): PRNGKey
        dW (Array): Wiener increments
        dt (Array): Time step size
        n (int, optional): Truncation of Fourier series. Defaults to 5.

    Returns:
        (Array, Array): Matrix of Ito integrals and Levy areas.


    NOTE: Based on https://github.com/mattja/sdeint/blob/master/sdeint/wiener.py#L102
    """

    dW = jnp.atleast_1d(dW)
    m = dW.shape[0]

    sqrt2h = jnp.sqrt(2.0 / dt)

    def body_fun(i, val):
        key, A0 = val
        next_key, key1, key2 = jax.random.split(key, 3)
        Xk = jax.random.normal(key1, shape=(m,))
        Yk = jax.random.normal(key2, shape=(m,))
        term1 = jnp.outer(Xk, (Yk + sqrt2h * dW))
        term2 = jnp.outer(Yk + sqrt2h * dW, Xk)
        A1 = A0 + (term1 - term2) / i

        return (next_key, A1)

    A0 = jnp.zeros((m, m))
    n = jax.lax.cond(m == 1, lambda _: 0, lambda _: n, None)  # No iteration for 1D
    init_val = (key, A0)
    _, A1 = jax.lax.fori_loop(1, n + 1, body_fun, init_val)

    A1 = (dt / (2.0 * jnp.pi)) * A1
    I = 0.5 * (jnp.outer(dW, dW) - dt * jnp.eye(m)) + A1

    return I, A1


def iterated_stratowich_integral_general(
    key: PRNGKey, dW: Array, dt: Array, n: int = 5
):
    """Matrix I approximating repeated Stratonovich integrals based on the method of Kloeden, Platen and Wright (1992)."""
    I, A = iterated_ito_integral_general(key, dW, dt, n)
    J = I + 0.5 * dt * jnp.eye(dW.shape[0])
    return J, A


def iterated_stochastic_integral_diagonal(key: PRNGKey, dW: Array, dt: Array, **kwargs):
    I_diag = 0.5 * (jnp.square(dW) - dt)
    return I_diag


def iterated_stochastic_integral_commutative_noise(
    key: PRNGKey, dW: Array, dt: Array, **kwargs
):
    I = jnp.outer(dW, dW) - dt * jnp.eye(dW.shape[0])
    return I


def get_iterated_integrals_fn(noise_type: str, sde_type: str):
    """Returns the iterated integrals function for a given noise type and sde type."""
    if noise_type == "diagonal":
        return iterated_stochastic_integral_diagonal
    elif noise_type == "commutative":
        return iterated_stochastic_integral_commutative_noise
    elif noise_type == "general":
        if sde_type == "ito":
            return lambda *args, **kwargs: iterated_ito_integral_general(
                *args, **kwargs
            )[0]
        elif sde_type == "stratonovich":
            return lambda *args, **kwargs: iterated_stratowich_integral_general(
                *args, **kwargs
            )[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


# Brownian bridge and tree


def brownian_path(key, x0, ts):
    shape = (ts.shape[0] - 1,) + x0.shape
    xs = x0[None, :] + jnp.cumsum(
        jnp.sqrt(ts[1] - ts[0]) * jax.random.normal(key, shape), axis=0
    )
    return jnp.concatenate([x0[None, :], xs], axis=0)


@jax.jit
def brownian_bridge(
    key: PRNGKey, t: Float, t0: Float, t1: Float, w0: Array, w1: Array
) -> Array:
    """Brownian bridge between two points.

    Args:
        key (PRNGKey): Random generator key.
        t (Float): Time at which to sample.
        t0 (Float): Time at which the bridge starts.
        t1 (Float): Time at which the bridge ends.
        w0 (Array): Value of the bridge at t0.
        w1 (Array): Value of the bridge at t1.

    Returns:
        Array: Value of the bridge at t.
    """
    length = t1 - t0
    dist_end = t1 - t
    dist_start = t - t0
    mean = (dist_end * w0 + dist_start * w1) / length
    std = jnp.sqrt(dist_end * dist_start / length)
    shape = mean.shape

    return mean + std * jrandom.normal(key, shape)


@jax.jit
def brownian_tree(
    key: PRNGKey, t: Float, t0: Float, t1: Float, w0: Array, tol: Float
) -> Array:
    """Brownian motion between two points using a tree. This allows to evaluate it at any time, without having to save the whole trajectory.

    Args:
        key (PRNGKeyArray): Random generator key.
        t (Float): Time at which to sample.
        t0 (Float): Start time.
        t1 (Float): End time.
        w0 (Array): Start value.
        tol (Float): Tolerance for the tree.

    Returns:
        Array: Value of the bridge at t.
    """
    key, init_key = jrandom.split(key, 2)
    shape = w0.shape

    t_half = t0 + 0.5 * (t1 - t0)
    w1 = jrandom.normal(init_key, shape) * jnp.sqrt(t1 - t0)
    w_half = brownian_bridge(key, t_half, t0, t1, w0, w1)

    init_state = (t0, t_half, t1, w0, w_half, w1, key)

    def cond_fun(state):
        start_time, _, end_time, _, _, _, _ = state
        return (end_time - start_time) > tol

    def body_fun(state):
        t0, t_half, t1, w0, w_half, w1, key = state

        _key1, _key2 = jrandom.split(key, 2)
        _cond = t > t_half
        _s = jnp.where(_cond, t_half, t0)
        _u = jnp.where(_cond, t1, t_half)
        _w_s = jnp.where(_cond, w_half, w0)
        _w_u = jnp.where(_cond, w1, w_half)
        _key = jnp.where(_cond, _key1, _key2)

        _t = _s + 0.5 * (_u - _s)
        _w_t = brownian_bridge(_key, _t, _s, _u, _w_s, _w_u)

        return (_s, _t, _u, _w_s, _w_t, _w_u, key)

    t0, t_half, t1, w0, w_half, w1, key = lax.while_loop(cond_fun, body_fun, init_state)

    rescale_t = (t - t0) / (t1 - t0)
    A = jnp.array([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])
    coeffs = jnp.tensordot(A, jnp.stack([w0, w_half, w1]), axes=1)
    return jnp.polyval(coeffs, rescale_t)


# Estimate weak and strong error
