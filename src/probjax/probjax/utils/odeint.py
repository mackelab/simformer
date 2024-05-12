import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax import lax
from jax import core
from jax.tree_util import tree_leaves
from jax.util import safe_map as map
from functools import partial
from jaxtyping import Array, Float, PyTree, Int
from typing import Callable, Optional


from probjax.core import custom_inverse
from probjax.utils.interpolation import linear_interpolation
from probjax.utils.solver import root
from probjax.utils.linalg import is_triangular_matrix
from probjax.utils.jaxutils import ravel_arg_fun, ravel_args


METHOD_STEP_FN = {}
METHOD_INFO = {}


def register_method(name: str, step_fn: Callable, info: Optional[dict] = None):
    """General method to register a step_fn for an ODE solver, thereby creating a new method.

    Args:
        name (str): Name of the method
        step_fn (Callable): Step function. This function should have the following signature:
            func(drift: Callable, t0: Array, y0: Array, f0: Array, dt: Array, **kwargs) -> Tuple[y1: Array, f1: Array, Tuple[y_error: Optional[Array], k: Optional[Array]]]
        info (Optional[dict], optional): Some information about your method. Defaults to None.

    Returns:
        _type_: _description_
    """
    METHOD_STEP_FN[name] = step_fn
    METHOD_INFO[name] = info
    return step_fn


def register_runge_kutta_method(
    name: str,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Optional[Array] = None,
    b_mid: Optional[Array] = None,
    info: Optional[str] = None,
    order: Optional[int] = None,
) -> Callable:
    """Register a Runge-Kutta method. This function will create a step function for the method and register it. A Runge-Kutta method is defined by the following equations:

    y1 = y0 + dt * sum_i b_sol[i] * k[i]
    k[i] = f(t0 + dt * c[i], y0 + dt * sum_j A[i,j] * k[j])

    and thus uniquely defined given a butcher tableau (c, A, b_sol).

    Args:
        name (str): Name of the method
        c (Array): The c vector of the butcher tableau. Time evaluation points.
        A (Array): The A matrix of the butcher tableau. Defines the coefficients of the evaluation points k.
        b_sol (Array): The b_sol vector of the butcher tableau. Defines the coefficients of the solution y1.
        b_error (Optional[Array], optional): How to compute an estimate of the local error. Defaults to None.
        info (Optional[str], optional): Information about the solver. Defaults to None.

    Returns:
        Callable: The step_fn of the method.
    """
    stages = len(c)
    if order is None:
        if stages >= 5:
            # Heuristic, should be provided on implementation.
            order = stages - 1
        else:
            order = stages

    assert jnp.all(c >= 0) and jnp.all(c <= 1), "c must be between 0 and 1"
    assert jnp.allclose(jnp.sum(b_sol), 1.0), "b_sol must sum to 1"

    assert A.shape == (
        stages,
        stages,
    ), f"Expected A.shape == ({stages}, {stages}), got {A.shape}"
    assert b_sol.shape == (
        stages,
    ), f"Expected b_sol.shape == ({stages},), got {b_sol.shape}"

    if jnp.allclose(A[-1], b_sol) and c[-1] == 1.0:
        last_equals_next = True
    else:
        last_equals_next = False

    if is_triangular_matrix(A) and jnp.diag(A).sum() == 0:
        explicit = True
    else:
        explicit = False

    if b_error is None:
        adaptive = False
    else:
        adaptive = True

    METHOD_INFO[name] = {
        "explicit": explicit,
        "order": order,
        "c": c,
        "A": A,
        "b_sol": b_sol,
        "b_error": b_error,
        "b_mid": b_mid,
        "interpolation_order": 4 if b_mid is not None else 3,
        "adaptive": adaptive,
        "info": info,
    }

    if explicit:
        step_fn = partial(
            explicit_runge_kutta_step,
            c=c,
            A=A,
            b_sol=b_sol,
            b_error=b_error,
            b_mid=b_mid,
            stages=stages,
            last_equals_next=last_equals_next,
        )
    else:
        step_fn = partial(
            implicit_runge_kutta_step,
            c=c,
            A=A,
            b_sol=b_sol,
            b_error=b_error,
            b_mid=b_mid,
            stages=order,
            last_equals_next=last_equals_next,
        )

    METHOD_STEP_FN[name] = step_fn
    return step_fn


def get_step_fn(method: str, dtype: Optional[Float] = None):
    """Returns the step function for a given method.

    Returns:
        Callable: Step function with corresponding method name.
    """
    step_fn = METHOD_STEP_FN[method]
    if dtype is not None:
        # Right numerical precision
        if hasattr(step_fn, "keywords"):
            step_fn.keywords["c"] = step_fn.keywords["c"].astype(dtype)
            step_fn.keywords["A"] = step_fn.keywords["A"].astype(dtype)
            step_fn.keywords["b_sol"] = step_fn.keywords["b_sol"].astype(dtype)
            if step_fn.keywords["b_error"] is not None:
                step_fn.keywords["b_error"] = step_fn.keywords["b_error"].astype(dtype)

    return step_fn


def get_method_info(method: str):
    """Returns the information about a given method."""
    return METHOD_INFO[method]


def get_methods():
    """Returns a list of all registered methods."""
    return list(METHOD_STEP_FN.keys())


@partial(jax.jit, static_argnums=(0, 10, 11))
def explicit_runge_kutta_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Array,
    b_mid: Array,
    stages: int,
    last_equals_next: bool,
):
    def body_fun(i, k):
        ti = t0 + dt * c[i]
        yi = y0 + dt * jnp.dot(A[i, :], k)
        ft = drift(ti, yi)
        return k.at[i, :].set(ft)

    k = jnp.zeros((stages, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = lax.fori_loop(1, stages + 1, body_fun, k)

    y1 = dt * jnp.dot(b_sol, k) + y0
    if last_equals_next:
        f1 = k[-1]
    else:
        f1 = drift(t0 + dt, y1)

    if b_error is None:
        y1_error = None
    else:
        y1_error = dt * jnp.dot(b_error, k)

    if b_mid is None:
        y1_mid = None
    else:
        y1_mid = dt * jnp.dot(b_mid, k) + y0

    return y1, f1, (y1_error, k, y1_mid)


def implicit_runge_kutta_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Array,
    b_mid: Optional[Array],
    stages: int = 2,
    **kwargs,
):
    ts = t0 + dt * c
    ts = ts.reshape(-1, 1)

    # Solve implicit equation
    def f(k):
        return k - dt * drift(y0 + jnp.dot(A, k), ts)

    # Uses root finding to solve implicit equation
    k0 = jnp.ones((stages, f0.shape[0]), f0.dtype) * f0
    k = root(f, k0)

    # Compute solution
    y1 = f0 * jnp.dot(b_sol, k) + y0
    f1 = lax.cond(c[-1] == 1.0, lambda _: k[-1], lambda _: drift(t0 + dt, y1), None)

    if b_error is None:
        y1_error = None
    else:
        y1_error = dt.astype(f0.dtype) * jnp.dot(b_error, k)

    if b_mid is None:
        y1_mid = None
    else:
        y1_mid = dt.astype(f0.dtype) * jnp.dot(b_mid, k) + y0

    return y1, f1, (y1_error, k, y1_mid)


def initial_step_size(
    drift: Callable,
    t0: Array,
    y0: Array,
    order: int,
    rtol: float,
    atol: float,
    f0: Array,
):
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    dtype = y0.dtype

    scale = atol + jnp.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
    d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
    y1 = y0 + h0.astype(dtype) * f0
    f1 = drift(t0 + h0, y1)
    d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

    h1 = jnp.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
    )

    return jnp.minimum(100.0 * h0, h1)


def mean_error_ratio(error_estimate, rtol, atol, y0, y1, norm: float = 2):
    err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
    err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
    return jnp.linalg.norm(err_ratio, ord=norm) / jnp.sqrt(len(err_ratio))


def optimal_step_size(
    last_step,
    mean_error_ratio,
    maxerror=1.0,
    safety=0.9,
    ifactor=10.0,
    dfactor=0.2,
    order=5.0,
):
    """Compute optimal Runge-Kutta stepsize."""
    dfactor = jnp.where(mean_error_ratio < maxerror, 1.0, dfactor)

    factor = jnp.minimum(
        ifactor, jnp.maximum(mean_error_ratio ** (-1.0 / order) * safety, dfactor)
    )
    return jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)

# TODO Wrap these paramters in functions ...
# Explicit Runge-Kutta methods
# https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods

# 1st order
# Euler's method
c = jnp.array([0])
A = jnp.array([[0]])
b_sol = jnp.array([1])
b_error = None
info = {
    "explicit": True,
    "order": 1,
    "c": c,
    "A": A,
    "b_sol": b_sol,
    "b_error": None,
    "info": "Euler's method",
    "adaptive": False,
}


# For efficiency, we use a custom implementation of Euler's method
@partial(jax.jit, static_argnums=(0,))
def _euler_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
):
    y1 = y0 + f0 * dt
    f1 = drift(t0 + dt, y1)
    return y1, f1, None


register_method("euler", _euler_step, info)

# 2nd order
# Heun's method
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = None
register_runge_kutta_method("heun", c, A, b_sol, b_error, info="Heun's method")

# Adaptive Heun's method
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = jnp.array([1.0, 0.0])
register_runge_kutta_method(
    "heun_euler", c, A, b_sol, b_error, info="Heun's Euler adaptive method"
)

# Midpoint method
c = jnp.array([0, 1 / 2])
A = jnp.array([[0, 0], [1 / 2, 0]])
b_sol = jnp.array([0, 1])
b_error = None
register_runge_kutta_method("midpoint", c, A, b_sol, b_error, info="Midpoint method")

# Ralston's method
c = jnp.array([0, 2 / 3])
A = jnp.array([[0, 0], [2 / 3, 0]])
b_sol = jnp.array([1 / 4, 3 / 4])
b_error = None
register_runge_kutta_method("ralston", c, A, b_sol, b_error, info="Ralston's method")


# 3rd order
# Kutta's third-order method
c = jnp.array([0, 1 / 2, 1])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
b_sol = jnp.array([1 / 6, 2 / 3, 1 / 6])
b_error = None
register_runge_kutta_method(
    "rk3", c, A, b_sol, b_error, info="Kutta's third-order method"
)

# Fehlberg's RK3(2) method (explicit) (adaptive)
c = jnp.array([0, 1 / 2, 1])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [1 / 256, 255 / 256, 0]])
b_sol = jnp.array([1 / 512, 255 / 256, 1 / 512])
b_error = jnp.array([1 / 256, 255 / 256, 0])
register_runge_kutta_method(
    "rk3(2)", c, A, b_sol, b_error, info="Fehlberg's RK3(2) method"
)

# Bosh3 method
c = jnp.array([0, 1 / 2, 3 / 4])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
b_sol = jnp.array([2 / 9, 1 / 3, 4 / 9])
b_error = None
register_runge_kutta_method("bosh3", c, A, b_sol, b_error, info="Bosh3 method")

# Heun's third-order method
c = jnp.array([0, 1 / 3, 2 / 3])
A = jnp.array([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]])
b_sol = jnp.array([1 / 4, 0, 3 / 4])
b_error = None
register_runge_kutta_method(
    "heun3", c, A, b_sol, b_error, info="Heun's third-order method"
)

# Van der Houwen's/Wray's method
c = jnp.array([0, 8 / 15, 2 / 3])
A = jnp.array([[0, 0, 0], [8 / 15, 0, 0], [1 / 4, 5 / 12, 0]])
b_sol = jnp.array([1 / 4, 0, 3 / 4])
b_error = None
register_runge_kutta_method(
    "vanderhouwen", c, A, b_sol, b_error, info="Van der Houwen's/Wray's method"
)

# Ralston's third-order method
c = jnp.array([0, 1 / 3, 2 / 3])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
b_sol = jnp.array([2 / 9, 1 / 3, 4 / 9])
b_error = None
register_runge_kutta_method(
    "ralston3", c, A, b_sol, b_error, info="Ralston's third-order method"
)

# Strong stability preserving Runge-Kutta methods of order 3
c = jnp.array([0, 1 / 2, 1])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [1 / 2, 1 / 2, 0]])
b_sol = jnp.array([1 / 6, 1 / 6, 2 / 3])
b_error = None
register_runge_kutta_method(
    "ssprk3",
    c,
    A,
    b_sol,
    b_error,
    info="Strong stability preserving Runge-Kutta methods of order 3",
)


# 4th order
# Classic Runge-Kutta method
c = jnp.array([0, 1 / 2, 1 / 2, 1])
A = jnp.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
b_sol = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
b_error = None
register_runge_kutta_method(
    "rk4", c, A, b_sol, b_error, info="Classic Runge-Kutta method"
)

# Bogacki-Shampine method, RK4(3) (explicit) (adaptive)
c = jnp.array([0, 1 / 2, 3 / 4, 1])
A = jnp.array(
    [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
)
b_sol = jnp.array([2 / 9, 1 / 3, 4 / 9, 0])
b_error = jnp.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
register_runge_kutta_method(
    "rk4(3)", c, A, b_sol, b_error, info="Bogacki-Shampine method"
)

# 3/8 rule
c = jnp.array([0, 1 / 3, 2 / 3, 1])
A = jnp.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
b_sol = jnp.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
b_error = None
register_runge_kutta_method("3/8", c, A, b_sol, b_error, info="3/8 rule")

# Ralston's method of order 4
c = jnp.array([0.0, 0.4, 0.45573725, 1.0])
A = jnp.array(
    [
        [0, 0, 0, 0],
        [0.4, 0, 0, 0],
        [0.29697761, 0.15875964, 0, 0],
        [0.21810040, -3.05096516, 3.83286476, 0],
    ]
)
b_sol = jnp.array([0.17476028, -0.55148066, 1.20553560, 0.17118478])
b_error = None
register_runge_kutta_method(
    "ralston4", c, A, b_sol, b_error, info="Ralston's method of order 4"
)

# 5th order

# Runge-Kutta method of order 5
c = jnp.array([0, 1 / 4, 1 / 4, 1 / 2, 3 / 4, 1])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0, 0],
        [1 / 8, 1 / 8, 0, 0, 0, 0],
        [0, 0, 1 / 2, 0, 0, 0],
        [3 / 16, -3 / 8, 3 / 8, 9 / 16, 0, 0],
        [-3 / 7, 8 / 7, 6 / 7, -12 / 7, 8 / 7, 0],
    ]
)
b_sol = jnp.array([7 / 90, 0, 32 / 90, 12 / 90, 32 / 90, 7 / 90])
b_error = None
register_runge_kutta_method(
    "rk5", c, A, b_sol, b_error, info="Runge-Kutta method of order 5"
)

# Fehlberg's RK5(4) method (explicit) (adaptive)
c = jnp.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0, 0],
        [3 / 32, 9 / 32, 0, 0, 0, 0],
        [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
        [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
    ]
)
b_sol = jnp.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
b_error = jnp.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
register_runge_kutta_method("rk5(4)", c, A, b_sol, b_error, info="RK5(4)")

# Cash-Karp method (explicit) (adaptive)
c = jnp.array([0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [3 / 10, -9 / 10, 6 / 5, 0, 0, 0],
        [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0, 0],
        [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096, 0],
    ]
)
b_sol = jnp.array([37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771])
b_error = jnp.array([2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4])
register_runge_kutta_method("cash-karp", c, A, b_sol, b_error, info="Cash-Karp method")

# 6th order
# Runge-Kutta method of order 6
c = jnp.array([0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 6, 0, 0, 0, 0, 0, 0],
        [1 / 12, 1 / 12, 0, 0, 0, 0, 0],
        [1 / 8, 0, 3 / 8, 0, 0, 0, 0],
        [91 / 500, -27 / 100, 78 / 125, 8 / 125, 0, 0, 0],
        [-11 / 20, 27 / 20, 12 / 5, -36 / 5, 5 / 2, 0, 0],
        [1 / 12, 0, 27 / 32, -4 / 3, 125 / 96, 5 / 48, 0],
    ]
)
b_sol = jnp.array([1 / 12, 0, 27 / 32, -4 / 3, 125 / 96, 5 / 48, 0])
b_error = None
register_runge_kutta_method(
    "rk6", c, A, b_sol, b_error, info="Runge-Kutta method of order 6"
)


# Dormand-Prince method
c = jnp.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ]
)
b_sol = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
b_error = jnp.array(
    [
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1.0 / 60.0,
    ]
)
b_mid = jnp.array(
    [
        6025192743 / 30085553152 / 2,
        0,
        51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2,
        187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2,
        11237099 / 235043384 / 2,
    ],
)
register_runge_kutta_method(
    "dopri5",
    c,
    A,
    b_sol,
    b_error,
    b_mid,
    info="Dormand-Prince method",
    order=5,
)


# Implicit Runge-Kutta methods
# https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Implicit_Runge%E2%80%93Kutta_methods

# 1st order
# Implicit Euler method
c = jnp.array([1.0])
A = jnp.array([[1.0]])
b_sol = jnp.array([1.0])
b_error = None
info = {
    "explicit": False,
    "order": 1,
    "c": c,
    "A": A,
    "b_sol": b_sol,
    "b_error": None,
    "info": "Implicit Euler method",
    "adaptive": False,
}


# For efficiency, we use a custom implementation of Euler's method
@partial(jax.jit, static_argnums=(0,))
def _implicit_euler_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
):
    def f(x):
        return x - y0 - f0 * dt

    y1 = root(f, y0)
    f1 = drift(t0 + dt, y1)
    return y1, f1, None


register_method("implicit_euler", _implicit_euler_step, info=info)


# 2nd order
# Implicit trapezoidal rule
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = None
register_runge_kutta_method(
    "implicit_trapezoidal", c, A, b_sol, b_error, info="Implicit trapezoidal rule"
)

# Implicit Crank-Nicolson method
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1 / 2, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = None
register_runge_kutta_method(
    "implicit_crank_nicolson",
    c,
    A,
    b_sol,
    b_error,
    info="Implicit Crank-Nicolson method",
)


# Exponential methods


def exponential_euler(drift, t0, y0, f0, dt):
    jacobian_fn = jax.jacfwd(drift, argnums=1)

    A = jacobian_fn(t0, y0)
    B = jnp.zeros_like(A)
    C = jnp.eye(A.shape[0])
    H = jnp.block([[A, C], [B, B]])
    eHdt = jax.scipy.linalg.expm(H * dt)
    phi0 = eHdt[0 : A.shape[0], 0 : A.shape[1]]
    phi1 = eHdt[0 : A.shape[0], A.shape[1] :]

    y1 = phi0 @ y0 + dt * phi1 @ (f0 - A @ y0)
    f1 = drift(t0 + dt, y1)

    return y1, f1, None


info = {
    "explicit": False,
    "order": 2,
    "info": "Exponential Euler method",
    "adaptive": False,
}

register_method("exp_euler", exponential_euler, info=info)


def _odeint_on_grid(drift: Callable, y0: Array, ts: Array, step_fn: Callable, scan_unroll:int=1):
    """Solve an ordinary differential equation discretized on a grid i.e. with fixed step size.

    Args:
        drift (Callable): Drift function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        step_fn (Callable): Step function.

    """
    # Time steps
    dts = ts[1:] - ts[:-1]

    def scan_fun(carry, data):
        t0, y0, f0 = carry
        t1, dt = data
        y1, f1, _ = step_fn(drift, t0, y0, f0, dt)
        return (t1, y1, f1), y1

    t0 = ts[0]
    f0 = drift(t0, y0)
    init_carry = (t0, y0, f0)
    _, ys = lax.scan(scan_fun, init_carry, (ts[1:], dts), unroll=scan_unroll)
    return jnp.concatenate((y0[None], ys))


def fit_1rd_order_polynomial(y0, y1, dy0, dy1, dt):
    a = (y1 - y0) * dt
    b = y0
    return a, b


def fit_3rd_order_polynomial(y0, y1, dy0, dy1, dt):
    """Be f(t) = a * t**3 + b * t**2 + c * t + d, then this function returns the coefficients a, b, c, d, which solve the system of equations:
    f(0) = y0
    f(1) = y1
    f'(0) = dy0
    f'(1) = dy1
    """
    d = y0
    c = dy0 * dt
    b = (-2 / 3 * c - d / dt + y1 / dt - dy1 / 3) * dt**2
    a = (y1 - b * dt**2 - c * dt - d) * dt**3

    return a, b, c, d


def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    """Be f(t) = a * t**4 + b * t**3 + c * t**2 + d * t + e, then this function returns the coefficients a, b, c, d, e, which solve the system of equations:
    f(0) = y0
    f(1) = y1
    f(1/2) = y_mid
    f'(0) = dy0
    f'(1) = dy1
    """
    a = -2.0 * dt * dy0 + 2.0 * dt * dy1 - 8.0 * y0 - 8.0 * y1 + 16.0 * y_mid
    b = 5.0 * dt * dy0 - 3.0 * dt * dy1 + 18.0 * y0 + 14.0 * y1 - 32.0 * y_mid
    c = -4.0 * dt * dy0 + dt * dy1 - 11.0 * y0 - 5.0 * y1 + 16.0 * y_mid
    d = dt * dy0
    e = y0
    return a, b, c, d, e


def interp_fit(y0, y1, f0, f1, dt, y_mid=None):
    if y_mid is not None:
        # We can use a 4th order polynomial (3 points and 2 gradients)
        return jnp.asarray(fit_4th_order_polynomial(y0, y1, y_mid, f0, f1, dt))
    else:
        # We can only use a 3rd order polynomial (2 points and 2 gradients)
        return jnp.asarray(fit_3rd_order_polynomial(y0, y1, f0, f1, dt))


def _odeint_adaptive(
    drift: Callable,
    y0: Array,
    ts: Array,
    step_fn: Callable,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    mxstep: int = jnp.inf,
    order: int = 5,
    dtinit: Optional[float] = None,
    dtmin: float = 0.0,
    dtmax: float = jnp.inf,
    maxerror: float = 1.0,
    safety: float = 0.9,
    ifactor: float = 10.0,
    dfactor: float = 0.2,
    error_norm: float = 2,
    interpolation_order: int = 4,
):
    def scan_fun(carry, target_t):
        def cond_fun(state):
            i, _, _, t, dt, _, _ = state
            return (t < target_t) & (i < mxstep) & (dt > 0)

        def body_fun(state):
            i, y, f, t, dt, last_t, interp_coeff = state
            # Predicts the next step
            next_y, next_f, (next_y_error, k, y_mid) = step_fn(drift, t, y, f, dt)
            next_t = t + dt
            # Error estimation and step size control
            error_ratio = mean_error_ratio(
                next_y_error, rtol, atol, y, next_y, error_norm
            )
            new_interp_coeff = interp_fit(y, next_y, f, next_f, dt, y_mid=y_mid)
            dt = jnp.clip(
                optimal_step_size(
                    dt,
                    error_ratio,
                    maxerror=maxerror,
                    safety=safety,
                    ifactor=ifactor,
                    dfactor=dfactor,
                    order=order,
                ),
                dtmin,
                dtmax,
            )

            new = [i + 1, next_y, next_f, next_t, dt, t, new_interp_coeff]
            old = [i + 1, y, f, t, dt, last_t, interp_coeff]
            return map(partial(jnp.where, error_ratio <= maxerror), new, old)

        _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
        _, _, t, _, last_t, interp_coeff = carry
        relative_output_time = (target_t - last_t) / (t - last_t)
        y_target = jnp.polyval(
            interp_coeff, relative_output_time.astype(interp_coeff.dtype)
        )
        return carry, y_target

    f0 = drift(ts[0], y0)
    if dtinit is None:
        dt = jnp.clip(
            initial_step_size(drift, ts[0], y0, 4, rtol, atol, f0),
            a_min=0.0,
            a_max=jnp.inf,
        )
    else:
        dt = dtinit
    interp_coeff = jnp.array([y0] * (interpolation_order + 1))
    init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff]
    _, ys = lax.scan(scan_fun, init_carry, ts[1:])
    return jnp.concatenate((y0[None], ys))


def _odeint(
    drift,
    y0: PyTree[Array],
    ts: Array,
    *args,
    method="rk4",
    dt: Optional[Float] = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    mxstep: int = jnp.inf,
    dtmin: float = 0.0,
    dtmax: float = jnp.inf,
    maxerror: float = 1.2,
    safety: float = 0.95,
    ifactor: float = 10.0,
    dfactor: float = 0.1,
    error_norm: float = 2,
    scan_unroll: int = 1,
):
    """Solve an ordinary differential equation.

    This function assumes that y0 is a single initial value, and that ts is a single time grid, with a single set of parameters.
    If you want to solve multiple ODEs, or multiple time grids, use vmap!

    Args:
        drift (Callable): Drift function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        args (Any): Additional arguments to pass to the drift function i.e. if it is parametrized.
        method (str, optional): Methods to use. Defaults to "euler".
        dt (Optional[Float], optional): Fixed step size. If it is an adaptive solver then this will be used as initializer. Defaults to None.
        rtol (float, optional): Relative tolerance (only relevant for adaptive solvers). Defaults to 1e-3.
        atol (float, optional): Absolute tolerance (only relevant for adaptive solvers). Defaults to 1e-3.
        mxstep (int, optional): Maximum number of steps (only relevant for adaptive solvers). Defaults to jnp.inf.
        dtmin (float, optional): Minimum step size (only relevant for adaptive solvers). Defaults to 0.0.
        dtmax (float, optional): Maximum step size (only relevant for adaptive solvers). Defaults to jnp.inf.
        maxerror (float, optional): Maximum error (only relevant for adaptive solvers). Defaults to 1.2.
        safety (float, optional): Safety factor (only relevant for adaptive solvers). Defaults to 0.95.
        ifactor (float, optional): Increase factor (only relevant for adaptive solvers). Defaults to 50.0.
        dfactor (float, optional): Decrease factor (only relevant for adaptive solvers). Defaults to 0.05.

    Returns:
        Array: Solution of the ODE.
    """
    # Flatten the initial value and time grid

    flat_y0, unravel = ravel_args(y0)

    y0 = jnp.atleast_1d(flat_y0)
    ts = jnp.atleast_1d(ts)

    # Consistent dtype, based on the initial value.
    dtype = y0.dtype
    ts = ts.astype(dtype)
    f = ravel_arg_fun(drift, unravel, 1)
    _f = lambda t, y: jnp.atleast_1d(f(t, y, *args)).astype(
        dtype
    )
    step_fn = get_step_fn(method, dtype=dtype)
    method_info = get_method_info(method)

    # Minimum step size, based on the dtype
    adaptive = method_info["adaptive"]

    # Solve the ODE
    if not adaptive:
        # Solvers without adaptive step size.
        if dt is None:
            # Use the provided time grid
            time_grid = ts
            ys = _odeint_on_grid(_f, y0, time_grid, step_fn, scan_unroll)
        else:
            # Use uniform time grid, with specified step size
            time_grid = jnp.arange(ts[0], ts[-1] + dt, dt)
            ys = _odeint_on_grid(_f, y0, time_grid, step_fn, scan_unroll)
            f_sol = jax.vmap(linear_interpolation(time_grid, ys))
            ys = f_sol(ts)
    else:
        # Solvers with adaptive step size.
        order = method_info["order"]
        interpolation_order = method_info["interpolation_order"]
        ys = _odeint_adaptive(
            _f,
            y0,
            ts,
            step_fn,
            rtol=rtol,
            atol=atol,
            mxstep=mxstep,
            order=order,
            dtinit=dt,
            dtmin=dtmin,
            dtmax=dtmax,
            maxerror=maxerror,
            safety=safety,
            ifactor=ifactor,
            dfactor=dfactor,
            error_norm=error_norm,
            interpolation_order=interpolation_order,
        )

    # Unflatten the solution
    ys = jax.vmap(unravel)(ys)
    return ys


# Inverse odeint
def _inv_odeint(drift, ys: Array, ts: Array, *args, **kwargs):
    y0 = ys[-1]
    xs = _odeint(drift, y0, ts[::-1], *args, **kwargs)
    return xs[-1]


# Ode and logabsdet
def _inv_logdet_odeint(drift, ys, ts, *args, **kwargs):
    _jac = jax.jacfwd(drift, argnums=1)
    jac = lambda t, x: jnp.atleast_2d(_jac(t, x))

    def aug_drift(t, state, *args):
        x, logdet = state
        dx = jnp.atleast_1d(drift(t, x, *args))
        dlogdet = jnp.atleast_1d(jnp.trace(jac(t, x)))
        return dx, dlogdet

    y0 = ys[-1]
    logdet0 = jnp.zeros(y0.shape[:-1])
    xs, logdets = _odeint(aug_drift, (y0, logdet0), ts[::-1], *args, **kwargs)

    return xs[-1], logdets[-1]


# ODEs are invertible, so we can define the inverse of the ODE solver
odeint = _odeint
odeint = custom_inverse(_odeint, static_argnums=(0,), inv_argnum=1)
odeint.definv(_inv_odeint)
odeint.definv_and_logdet(_inv_logdet_odeint)
