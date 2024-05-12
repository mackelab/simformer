import jax
from jax import numpy as jnp
from jax import lax

from jaxtyping import Array, Float, Int
from typing import Callable, Optional


def linear_interpolation(ts: Array, ys: Array) -> Callable[[Float], Array]:
    """Linear interpolation function for a given set of points (ts, ys). Here ts must be a one dimensional sorted array and ys can be any array with the same length as ts on axis 0.
        Outside of the data range, the function returns the value of the nearest data point.

    Args:
        ts (Array): Time points
        ys (Array): Values at time points

    Returns:
        Callable[[Float], Array]: Interpolation function that can be evaluated at any time point.
    """

    shape = ys.shape
    event_shape = ys.shape[1:]
    ys = ys.reshape(shape[0], -1)

    def interpolate(t: Float) -> Array:
        return jax.vmap(jnp.interp, in_axes=(None, None, -1))(t, ts, ys).reshape(
            event_shape
        )

    return interpolate


def polynomial_interpolation(
    ts: Array, ys: Array, degree: Int = 3, window: Int = None
) -> Callable[[Float], Array]:
    """Polynomial interpolation function for a given set of points (ts, ys). Here ts must be a one dimensional sorted array and ys can be any array with the same length as ts on axis 0.
        The interpolation is done using a polynomial of degree 'degree'. The window parameter can be used to limit the range of data points used for interpolation. If window is None, the interpolation is done using all the data points.

        Note: Outside of the data range, the function does return the value of the interpolant.

    Args:
        ts (Array): _description_
        ys (Array): _description_
        degree (Int, optional): _description_. Defaults to 3.
        window (Optional[Int], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    shape = ys.shape
    event_shape = ys.shape[1:]
    ys = ys.reshape(shape[0], -1)

    if window is None:
        window = degree // 2
    else:
        window = window

    def interpolate(t: Float) -> Array:
        index = jnp.searchsorted(ts, t)
        index = lax.cond(index - window < 0, lambda: window, lambda: index)
        index = lax.cond(
            index + window > len(ts), lambda: len(ts) - window, lambda: index
        )
        indices = index + jnp.arange(-window, window + 1)
        data_x = ts[indices]
        data_y = ys[indices, :]
        p = jnp.polyfit(data_x, data_y, degree)
        return jnp.polyval(p, t).reshape(event_shape)

    return interpolate
