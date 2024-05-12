from typing import Tuple

import jax
import jax.numpy as jnp

from jax import Array

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from functools import partial


def _normalize_knot_slopes(
    unnormalized_knot_slopes: Array, min_knot_slope: float
) -> Array:
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    if min_knot_slope >= 1.0:
        raise ValueError(
            f"The minimum knot slope must be less than 1; got" f" {min_knot_slope}."
        )
    min_knot_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope


def _rational_quadratic_spline_fwd(
    x: Array, x_pos: Array, y_pos: Array, knot_slopes: Array
) -> Tuple[Array, Array]:
    """Applies a rational-quadratic spline to a scalar.

    Args:
      x: a scalar (0-dimensional array). The scalar `x` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[x_pos[0], x_pos[-1]]`, and it will be transformed linearly if it's
        outside that interval.
      x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
      y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
      knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
    Returns:
      A tuple of two scalars: the output of the transformation and the log of the
      absolute first derivative at `x`.
    """
    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)
    # If y does not fall into any bin, we use the first spline in the following
    # computations to avoid numerical issues.
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    # Dot product of each parameter with the correct bin mask.
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width

    z = (x - x_pos_bin[0]) / bin_width
    # `z` should be in range [0, 1] to avoid NaNs later. This can happen because
    # of small floating point issues or when x is outside of the range of bins.
    # To avoid all problems, we restrict z in [0, 1].
    z = jnp.clip(z, 0.0, 1.0)
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    numerator = bin_height * (bin_slope * sq_z + knot_slopes_bin[0] * z1mz)
    denominator = bin_slope + slopes_term * z1mz
    y = y_pos_bin[0] + numerator / denominator

    # Compute log det Jacobian.
    # The logdet is a sum of 3 logs. It is easy to see that the inputs of the
    # first two logs are guaranteed to be positive because we ensured that z is in
    # [0, 1]. This is also true of the log(denominator) because:
    # denominator
    # == bin_slope + (knot_slopes_bin[1] + knot_slopes_bin[0] - 2 * bin_slope) *
    # z*(1-z)
    # >= bin_slope - 2 * bin_slope * z * (1-z)
    # >= bin_slope - 2 * bin_slope * (1/4)
    # == bin_slope / 2
    logdet = (
        2.0 * jnp.log(bin_slope)
        + jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        - 2.0 * jnp.log(denominator)
    )

    # If x is outside the spline range, we default to a linear transformation.
    y = jnp.where(below_range, (x - x_pos[0]) * knot_slopes[0] + y_pos[0], y)
    y = jnp.where(above_range, (x - x_pos[-1]) * knot_slopes[-1] + y_pos[-1], y)
    logdet = jnp.where(below_range, jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, jnp.log(knot_slopes[-1]), logdet)
    return y, logdet


def _safe_quadratic_root(a: Array, b: Array, c: Array) -> Array:
    """Implement a numerically stable version of the quadratic formula."""
    # This is not a general solution to the quadratic equation, as it assumes
    # b ** 2 - 4. * a * c is known a priori to be positive (and which of the two
    # roots is to be used, see https://arxiv.org/abs/1906.04032).
    # There are two sources of instability:
    # (a) When b ** 2 - 4. * a * c -> 0, sqrt gives NaNs in gradient.
    # We clip sqrt_diff to have the smallest float number.
    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))
    # If sqrt_diff is non-positive, we set sqrt to 0. as it should be positive.
    safe_sqrt = jnp.where(sqrt_diff > 0.0, safe_sqrt, 0.0)
    # (b) When 4. * a * c -> 0. We use the more stable quadratic solution
    # depending on the sign of b.
    # See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf (eq 7 and 8).
    # Solution when b >= 0
    numerator_1 = 2.0 * c
    denominator_1 = -b - safe_sqrt
    # Solution when b < 0
    numerator_2 = -b + safe_sqrt
    denominator_2 = 2 * a
    # Choose the numerically stable solution.
    numerator = jnp.where(b >= 0, numerator_1, numerator_2)
    denominator = jnp.where(b >= 0, denominator_1, denominator_2)
    return numerator / denominator


def _rational_quadratic_spline_inv(
    y: Array, x_pos: Array, y_pos: Array, knot_slopes: Array
) -> Tuple[Array, Array]:
    """Applies the inverse of a rational-quadratic spline to a scalar.

    Args:
      y: a scalar (0-dimensional array). The scalar `y` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[y_pos[0], y_pos[-1]]`, and it will be transformed linearly if it's
        outside that interval.
      x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
      y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
      knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
    Returns:
      A tuple of two scalars: the output of the inverse transformation and the log
      of the absolute first derivative of the inverse at `y`.
    """
    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)
    # If y does not fall into any bin, we use the first spline in the following
    # computations to avoid numerical issues.
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    # Dot product of each parameter with the correct bin mask.
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    # These are the parameters for the corresponding bin.
    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width
    w = (y - y_pos_bin[0]) / bin_height
    w = jnp.clip(w, 0.0, 1.0)  # Ensure w is in [0, 1].
    # Compute quadratic coefficients: az^2 + bz + c = 0
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    c = -bin_slope * w
    b = knot_slopes_bin[0] - slopes_term * w
    a = bin_slope - b

    # Solve quadratic to obtain z and then x.
    z = _safe_quadratic_root(a, b, c)
    z = jnp.clip(z, 0.0, 1.0)  # Ensure z is in [0, 1].
    x = bin_width * z + x_pos_bin[0]

    # Compute log det Jacobian.
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet = (
        -2.0 * jnp.log(bin_slope)
        - jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        + 2.0 * jnp.log(denominator)
    )

    # If y is outside the spline range, we default to a linear transformation.
    x = jnp.where(below_range, (y - y_pos[0]) / knot_slopes[0] + x_pos[0], x)
    x = jnp.where(above_range, (y - y_pos[-1]) / knot_slopes[-1] + x_pos[-1], x)
    logdet = jnp.where(below_range, -jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, -jnp.log(knot_slopes[-1]), logdet)
    return x, logdet


@partial(custom_inverse, inv_argnum=1)
def rational_quadratic_spline(
    params: Array,
    x: Array,
    range_min_x: float = -1.0,
    range_max_x: float = 1.0,
    range_min_y: float = -1.0,
    range_max_y: float = 1.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)

    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        range_max_x - range_min_x
    ) + range_min_x
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        range_max_y - range_min_y
    ) + range_min_y

    y, _ = _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)
    return y


def inv_rational_quadratic_spline(
    params: Array,
    x: Array,
    range_min_x=-1.0,
    range_max_x=1.0,
    range_min_y=-1.0,
    range_max_y=1.0,
    min_bin_size=1e-4,
    min_knot_slope: float = 1e-4,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)

    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        range_max_x - range_min_x
    ) + range_min_x
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        range_max_y - range_min_y
    ) + range_min_y
    y, log_det = _rational_quadratic_spline_inv(x, x_pos, y_pos, knot_slopes)
    return y, jnp.squeeze(log_det)


rational_quadratic_spline.definv_and_logdet(inv_rational_quadratic_spline)

from probjax.utils.solver import root_scalar


@partial(custom_inverse, inv_argnum=1)
def learnable_mixture_cdf(
    params: Array,
    y: Array,
    min_value=-10.0,
    max_value=10.0,
    **kwargs,
):
    def f(x):
        return _inv_learnable_mixture_cdf(params, x) - y

    x = root_scalar(
        f,
        bracket=(min_value * jnp.ones_like(y), max_value * jnp.ones_like(y)),
        **kwargs,
    )

    return x


def _inv_learnable_mixture_cdf(
    params: Array,
    x: Array,
    **kwargs,
):
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    scale = jnp.exp(scale)
    x_ks = (x[..., None] - loc) / scale
    cdf = jnp.mean(jax.nn.sigmoid(x_ks), -1)
    out = jax.scipy.stats.norm.ppf(cdf)
    return out


def _inv_and_logdet_learnable_mixture_cdf(params, x, **kwargs):
    _f = jax.vmap(jax.value_and_grad(_inv_learnable_mixture_cdf, argnums=1))
    value, grad = _f(params, x)
    return value, jnp.log(jnp.abs(grad))


learnable_mixture_cdf.definv(_inv_learnable_mixture_cdf)
learnable_mixture_cdf.definv_and_logdet(_inv_and_logdet_learnable_mixture_cdf)
