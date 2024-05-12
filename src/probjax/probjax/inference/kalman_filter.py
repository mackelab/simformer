from ast import Call
import jax
import jax.numpy as jnp
from jax import lax
from jax.random import PRNGKeyArray

from typing import Callable, Tuple, Optional
from jaxtyping import Array, Float

from probjax.utils.linalg import matrix_fraction_decomposition, transition_matrix

from functools import partial


def predict_discrete(
    drift_matrix: Callable,
    diffusion_matrix: Callable,
    t0: Float,
    t1: Float,
    mu0: Array,
    cov0: Array,
) -> Tuple[Array, Array]:
    """Predicts the state at t1 given the state t0, for a model defined by the drift and diffusion matrix.

    Args:
        t0 (Float): Start time
        t1 (Float): End time
        mu0 (Array): Mean at t0
        cov0 (Array): Covariance at t0
        drift_matrix (Callable): Drift matrix
        diffusion_matrix (Callable): Diffusion matrix

    Returns:
        Tuple[Array, Array]: Mean and covariance at t1
    """
    Phi, Q = matrix_fraction_decomposition(
        t0, t1, drift_matrix(t0, mu0), diffusion_matrix(t0, mu0)
    )
    mu1 = jnp.dot(Phi, mu0)
    cov1 = jnp.dot(Phi, jnp.dot(cov0, Phi.T)) + Q
    return mu1, cov1


def update(
    mu0: Array, cov0: Array, t_o: Float, y_o: Float, C_o: Array, R_o: Array
) -> Tuple[Array, Array]:
    """Updates the state given a new observation at time t_o with value y_o.

    Args:
        mu0 (Array): Predicted mean
        cov0 (Array): Predicted covariance
        t_o (Float): Time of observation
        y_o (Float): Value of observation
        C_o (Array): Measurement projection matrix.
        R_o (Array): Measurement noise covariance matrix.

    Returns:
        Tuple[Array, Array]: Updated mean and covariance
    """

    if C_o is None:
        C_o = jnp.eye(mu0.shape[0])

    y = y_o
    m = C_o @ mu0
    r = y - m
    S = C_o @ cov0 @ C_o.T
    if R_o is not None:
        S = S + R_o
    K = cov0 @ jnp.linalg.solve(S, C_o).T

    mu1 = mu0 + K @ r
    cov1 = cov0 - K @ S @ K.T

    return mu1, cov1


def smooth_discrete(
    drift_matrix: Callable,
    t0: Float,
    t1: Float,
    mu0_s: Array,
    cov0_s: Array,
    mu0: Array,
    cov0: Array,
    mu0_: Array,
    cov0_: Array,
) -> Tuple[Array, Array]:
    """Discrete time Rauch-Tung-Striebel smoothing.

    Args:
        t0 (Float): Time of start
        t1 (Float): Time of end
        mu0_s (Array): Smoothed mean at t0
        cov0_s (Array): Smoothed covariance at t0
        mu0 (Array): Unsmoothed mean at t0
        cov0 (Array): Unsommthed covariance at t0
        mu0_ (Array): Prediction mean at t0
        cov0_ (Array): Prediction covariance at t0
        drift_matrix (Callable): Drift matrix

    Returns:
        Tuple[Array, Array]: Updated mean and covariance.
    """
    Phi = transition_matrix(drift_matrix(t0, mu0_s), t0, t1)
    G = jnp.dot(cov0, jnp.linalg.solve(cov0_, Phi).T)
    mu1 = mu0 + jnp.dot(G, mu0_s - jnp.dot(Phi, mu0_))
    cov1 = cov0 + jnp.dot(G, jnp.dot(cov0_s - cov0_, G.T))
    return mu1, cov1


def get_prediction_step(drift, diffusion, method="mfd_linearized"):
    """Returns a prediction function for a model defined by the drift and diffusion matrix.

    Args:
        drift_matrix (Callable): Drift matrix
        diffusion_matrix (Callable): Diffusion matrix

    Returns:
        Callable: Prediction function
    """
    if method == "mfd_linearized":
        drift_matrix = lambda t, x: jax.jacfwd(drift, argnums=1)(t, x)
        return partial(predict_discrete, drift_matrix, diffusion)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")


def get_update_step(C_o, R_o, method="kalman"):
    """Returns an update function for a model defined by the drift and diffusion matrix.

    Args:
        measurement (Callable): Measurement function

    Returns:
        Callable: Update function
    """
    if method == "kalman":
        return partial(update, C_o=C_o, R_o=R_o)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")


@partial(jax.jit, static_argnums=(0,1, 8,9))
def filter(
    drift: Callable,
    diffusion: Callable,
    ts: Array,
    mu0: Array,
    cov0: Array,
    t_o: Optional[Array] = None,
    y_o: Optional[Array] = None,
    C_o: Optional[Array] = None,
    R_o: Optional[Array] = None,
    return_mu_cov_cache: bool = False,
    prediction_method="mfd_linearized",
) -> Tuple[Array, Array, Array, Array]:
    """Kalman filter for a discrete time model on time grid ts.

    Args:
        ts (Array): Time grid to filter on.
        mu0 (Array): Starting mean
        cov0 (Array): Starting covariance
        predict (Callable): Prediction function
        update (Callable): Update function
        t_o (Array): Observation times
        y_o (Array): Observation values
        C_o (Array): Observation projection matrix
        R_o (Array): Observation noise covariance matrix

    Returns:
        Tuple[Array, Array, Array, Array]: Mean and covs at observation times, and mean and covs at all times.
    """

    mu0 = jnp.atleast_1d(mu0)
    cov0 = jnp.atleast_2d(cov0)
    ts = jnp.atleast_1d(ts)
    d = mu0.shape[0]
    dtype = mu0.dtype
    # Check consistent shapes
    assert (
        mu0.shape[0] == cov0.shape[0] == cov0.shape[1]
    ), "Dimension mismatch, cov0 must be a square matrix of shape (d,d) and mu0 must be a vector of shape (d,)"

    # Consistent dtype
    cov0 = cov0.astype(dtype)
    ts = ts.astype(dtype)
    _drift = lambda t, x: jnp.atleast_1d(drift(t, x)).astype(dtype)
    _diffusion = lambda t, x: jnp.atleast_2d(diffusion(t, x)).astype(dtype)

    if t_o is not None and y_o is not None:
        # Merge observation times with time grid
        index = jnp.searchsorted(ts, t_o)
        ts_merged = jnp.insert(ts, index, t_o)
        index = jnp.searchsorted(ts_merged, t_o)

        mask_array = jnp.zeros(len(ts_merged), dtype=jnp.int32)
        index_array = mask_array.at[index].set(1).cumsum()
        num_obs = len(t_o)
    else:
        ts_merged = ts
        index_array = jnp.zeros(len(ts), dtype=jnp.int32)
        t_o = jnp.zeros((1,)) * jnp.nan
        y_o = jnp.zeros((1,)) * jnp.nan
        num_obs = 0


    predict = get_prediction_step(_drift, _diffusion, method="mfd_linearized")
    update = get_update_step(C_o, R_o) 

    _identity = lambda mu, cov, *args:  (mu, cov)

    def scan_fun(carry, data):
        mu0, cov0, t0, mu_cache, cov_cache = carry
        t1, j = data
        mu1_, cov1_ = predict(t0, t1, mu0, cov0)
        is_update = t_o[j] == t1
        mu1, cov1 = lax.cond(
            is_update, update, _identity, mu1_, cov1_, t_o[j], y_o[j]
        )
        if return_mu_cov_cache:
            mu_cache, cov_cache = lax.cond(is_update, lambda m,c,j: (m.at[j].set(mu1_), c.at[j].set(cov1_)), lambda m,c,j: (m,c), mu_cache, cov_cache, j)

        return (mu1, cov1, t1, mu_cache, cov_cache), (mu1, cov1)

    init_carry = (mu0, cov0, ts[0], jnp.zeros((num_obs,  d)), jnp.zeros((num_obs, d, d)))
    final_carry, (mus, covs) = lax.scan(
        scan_fun, init_carry, (ts_merged[1:], index_array[:-1])
    )

    mus = jnp.concatenate([mu0[None], mus])
    covs = jnp.concatenate([cov0[None], covs])

    if return_mu_cov_cache:
        return mus, covs, final_carry[-2], final_carry[-1]
    else:
        return mus, covs


def smooth(
    ts: Array, mus: Array, covs: Array, mus_: Array, covs_: Array, smooth: Callable
) -> Tuple[Array, Array]:
    """Smooths the state given a Kalman filter output.

    Args:
        ts (Array): Time grid
        mus (Array): Means
        covs (Array): Covs
        mus_ (Array): Predicted means
        covs_ (Array): Predicted covs
        smooth (Callable): Smoothing function

    Returns:
        Tuple[Array, Array]: _description_
    """

    idx_last = jnp.where((mus != mus_).all(-1))[-1][-1]
    mus_needed_ = jnp.flip(mus_[1 : idx_last + 1])
    covs_needed_ = jnp.flip(covs_[1 : idx_last + 1])
    mus_needed = jnp.flip(mus[:idx_last])
    covs_needed = jnp.flip(covs[:idx_last])
    ts_needed = jnp.flip(ts[:idx_last])

    def scan_fun(carry, data):
        (mu0_s, cov0_s, t1) = carry
        t0, mu0, cov0, mu0_, cov0_ = data
        mu1, cov1 = smooth(t0, t1, mu0_s, cov0_s, mu0, cov0, mu0_, cov0_)
        return (mu1, cov1, t0), (mu1, cov1)

    init_carry = (mus[idx_last], covs[idx_last], ts[idx_last])
    _, (mus_s, covs_s) = lax.scan(
        scan_fun,
        init_carry,
        (ts_needed, mus_needed, covs_needed, mus_needed_, covs_needed_),
    )

    mus = jnp.concatenate([mus_s[::-1], mus[idx_last:]])
    covs = jnp.concatenate([covs_s[::-1], covs[idx_last:]])

    return mus, covs
