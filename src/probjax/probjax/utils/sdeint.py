import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax import lax
from jax import core
from jax.tree_util import tree_leaves

from functools import partial
from jaxtyping import Array, Float, PyTree, Int
from typing import Callable, Optional, Union
from jax.random import PRNGKeyArray


from probjax.utils.brownian import get_iterated_integrals_fn
from probjax.utils.linalg import is_matrix, is_triangular_matrix


METHOD_STEP_FN = {}
METHOD_INFO = {}


def register_method(name: str, func: Callable, info: Optional[dict] = None) -> Callable:
    """This function registers a general step_fn for a method.

    Args:
        name (str): Name of the method.
        func (Callable): Step function. The function must have the following signature:
            def step_fn(
                drift: Callable,
                diffusion: Callable,
                t0: Array,
                y0: Array,
                f0: Array,
                g0: Array,
                dt: Array,
                dWt: Array,
                dWtdWs: Array,
            )
        info (Optional[dict], optional): Additional information about the method. Defaults to None.

    Returns:
        Callable: step_fn
    """
    METHOD_STEP_FN[name] = func
    METHOD_INFO[name] = info
    return func


def register_stochastic_runge_kutta_method(
    name: str,
    c0: Array,
    c1: Array,
    A0: Array,
    A1: Array,
    B0: Array,
    B1: Array,
    b_sol: Array,
    gamma0: Array,
    gamma1: Array,
    b_error: Optional[Array] = None,
    order: Optional[int] = None,
    strong_order: Optional[int] = None,
    weak_order: Optional[int] = None,
):
    order = len(c0)  # This is wrong
    stages = len(c0)

    # TODO: Check if A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error are valid
    # TODO: Check if order, strong_order, weak_order are valid
    if is_triangular_matrix(A0) and jnp.diag(A1).sum() == 0:
        explicit = True
    else:
        explicit = False

    if b_error is None:
        adaptive = False
    else:
        adaptive = True

    METHOD_INFO[name] = {
        "order": order,  # This is the deterministic order
        "strong_order": strong_order,  # Stochastic strong order
        "weak_order": weak_order,  # Stochastic weak order
        "stages": stages,
        "A0": A0,
        "A1": A1,
        "B0": B0,
        "B1": B1,
        "b_sol": b_sol,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "b_error": b_error,
        "adaptive": adaptive,
        "requires_iterated_integrals": True,
    }

    if explicit:
        step_fn = partial(
            explicit_stochastic_runge_kutta_step,
            c0=c0,
            c1=c1,
            A0=A0,
            A1=A1,
            B0=B0,
            B1=B1,
            b_sol=b_sol,
            gamma0=gamma0,
            gamma1=gamma1,
            b_error=b_error,
            stages=stages,
        )
    else:
        raise NotImplementedError

    METHOD_STEP_FN[name] = step_fn

    return step_fn


def get_step_fn(
    method: str, dtype: Optional[Float] = None, sde_type: Optional[str] = None
) -> Callable:
    """Returns the step function for a given method.

    Returns:
        Callable: Step function with corresponding method name.
    """
    step_fn = METHOD_STEP_FN[method]

    if dtype is not None:
        # Right numerical precision
        if hasattr(step_fn, "keywords"):
            step_fn.keywords["c0"] = step_fn.keywords["c0"].astype(dtype)
            step_fn.keywords["c1"] = step_fn.keywords["c1"].astype(dtype)
            step_fn.keywords["A0"] = step_fn.keywords["A0"].astype(dtype)
            step_fn.keywords["A1"] = step_fn.keywords["A1"].astype(dtype)
            step_fn.keywords["B0"] = step_fn.keywords["B0"].astype(dtype)
            step_fn.keywords["B1"] = step_fn.keywords["B1"].astype(dtype)
            step_fn.keywords["gamma0"] = step_fn.keywords["gamma0"].astype(dtype)
            step_fn.keywords["gamma1"] = step_fn.keywords["gamma1"].astype(dtype)
            step_fn.keywords["b_sol"] = step_fn.keywords["b_sol"].astype(dtype)
            if step_fn.keywords["b_error"] is not None:
                step_fn.keywords["b_error"] = step_fn.keywords["b_error"].astype(dtype)

    # sde_type

    return step_fn


def get_method_info(method: str) -> Callable:
    """Returns the info for a given method."""
    return METHOD_INFO[method]


def get_methods():
    """Returns the list of available methods."""
    return list(METHOD_STEP_FN.keys())


@partial(jax.jit, static_argnums=(0, 1, 19, 20))
def explicit_stochastic_runge_kutta_step(
    drift: Callable,
    diffusion: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    g0: Array,
    dt: Array,
    dWt: Array,
    dWtdWs: Array,
    c0: Array,
    c1: Array,
    A0: Array,
    A1: Array,
    B0: Array,
    B1: Array,
    b_sol: Array,
    gamma0: Array,
    gamma1: Array,
    b_error: Array,
    stages: int,
    is_diagonal: bool = False,
    *kwargs,
):
    """Explicit stochastic Runge-Kutta method.

    Paper: https://preprint.math.uni-hamburg.de/public/papers/prst/prst2010-02.pdf

    Args:
        drift (Callable): _description_
        diffusion (Callable): _description_
        t0 (Array): _description_
        y0 (Array): _description_
        f0 (Array): _description_
        g0 (Array): _description_
        dt (Array): _description_
        dWt (Array): _description_
        dWtdWs (Array): _description_
        c0 (Array): _description_
        c1 (Array): _description_
        A0 (Array): _description_
        A1 (Array): _description_
        B0 (Array): _description_
        B1 (Array): _description_
        b_sol (Array): _description_
        gamma0 (Array): _description_
        gamma1 (Array): _description_
        b_error (Array): _description_
        order (int): _description_
        is_diagonal (bool, optional): _description_. Defaults to False.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    dtsqrt = jnp.sqrt(jnp.abs(dt))
    dtsqrt_vec = jnp.ones_like(dWt) * dtsqrt
    m = dWt.shape[0]
    d = y0.shape[0]

    if is_diagonal:
        reduction_dWt = "s, smi, j -> i"
    else:
        # General case not working yet ...
        reduction_dWt = (
            "s, smij, j -> i"  # Average drift evaluation over s, then matmul with dWt
        )
    diffusion_vec = jax.vmap(diffusion, in_axes=(None, 0))  # Vectorize diffusion

    def body_fun(i, data):
        k1, k2 = data
        ti1 = t0 + dt * c0[i]
        ti2 = t0 + dt * c1[i]

        yi1 = (
            y0
            + jnp.dot(A0[i, :], k1) * dt
            + 1/d * jnp.einsum(reduction_dWt, B0[i, :], k2, dWt)
        )

        yi2 = y0 + jnp.dot(A1[i, :], k1) * dt

        yi2 = jnp.broadcast_to(yi2, (m,) + yi2.shape)

    

        for k in range(m):
            
            res = jnp.einsum(
                reduction_dWt, B1[i, :], k2, jnp.atleast_1d(dWtdWs[k, ...])
            ) / jnp.sqrt(dt)
            yi2 = yi2.at[k, ...].add(res)


        ft = drift(ti1, yi1)
        gt = diffusion_vec(ti2, yi2)
        return k1.at[i, ...].set(ft), k2.at[i, ...].set(gt)

    # Drift evaluations at support points
    k1 = jnp.zeros((stages,) + f0.shape, f0.dtype).at[0, :].set(f0)
    # Diffusion evaluations at support points
    k2 = (
        jnp.zeros((stages, m) + g0.shape, g0.dtype).at[0, :].set(g0)
    )  # Diffusion evaluations at support points

    k1, k2 = lax.fori_loop(1, stages + 1, body_fun, (k1, k2))

    y1 = (
        y0
        + jnp.dot(b_sol, k1) * dt
        + 1/d*jnp.einsum(reduction_dWt, gamma0, k2, dWt)
        + 1/d*jnp.einsum(reduction_dWt, gamma1, k2, dtsqrt_vec)
    )

    f1 = drift(t0 + dt, y1)
    g1 = diffusion(t0 + dt, y1)

    if b_error is None:
        y1_error = None
    else:
        raise NotImplementedError

    return y1, f1, g1, (y1_error, k1, k2)


# Strong order 0.5 methods


# Euler-Maruyama method
# We use a custom implementation to avoid the overhead of the general method
@partial(jax.jit, static_argnums=(0, 1, 9))
def _euler_maruyama_step_fn(
    drift: Callable,
    diffusion: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    g0: Array,
    dt: Array,
    dWt: Array,
    dWtdWs: Union[Array, None],
    is_diagonal: bool = False,
    **kwargs,
):
    if is_diagonal:
        reduction = "i,i -> i"
    else:
        reduction = "ij, j -> i"

    y1 = y0 + dt * f0 + jnp.einsum(reduction, g0, dWt)
    f1 = drift(t0 + dt, y1)
    g1 = diffusion(t0 + dt, y1)
    return y1, f1, g1, None


info = {
    "order": 1,
    "strong_order": 0.5,
    "weak_order": 1,
    "A0": jnp.array([[0.0]]),
    "A1": jnp.array([[0.0]]),
    "B0": jnp.array([[0.0]]),
    "B1": jnp.array([[0.0]]),
    "b_sol": jnp.array([1.0]),
    "gamma0": jnp.array([1.0]),
    "gamma1": jnp.array([1.0]),
    "b_error": None,
    "adaptive": False,
}
register_method("euler_maruyama", _euler_maruyama_step_fn, info)


# Strong order 1.0 method


@partial(jax.jit, static_argnums=(0, 1, 9))
def _milstein_step_fn(
    drift: Callable,
    diffusion: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    g0: Array,
    dt: Array,
    dWt: Array,
    dWtdWs: Array,
    is_diagonal: bool = False,
    **kwargs,
):
    _g_jac = jax.jacfwd(lambda t, x: diffusion(t, x).sum(0), argnums=1)

    if is_diagonal:
        reduction1 = "i,i -> i"
        reduction2 = "i,i,i -> i"
    else:
        reduction1 = "ij, j -> i"
        reduction2 = "nm, mm, mn -> n"

    g0_grad = _g_jac(t0, y0)

    y1 = (
        y0
        + dt * f0
        + jnp.einsum(reduction1, g0, dWt)  # g(t0, y0) * dWt
        + jnp.einsum(
            reduction2, g0_grad.T, dWtdWs, g0
        )  # g_t(t0, y0) * g'_s(t0, y0) * dWt * dWs
    )

    f1 = drift(t0 + dt, y1)
    g1 = diffusion(t0 + dt, y1)
    return y1, f1, g1, None


info = {
    "order": 1,
    "strong_order": 1,
    "weak_order": 1,
    "adaptive": False,
    "requires_iterated_integrals": True,
}
register_method("milstein", _milstein_step_fn, info)


# Strong order 1.0 methods

# SRI1
c0 = jnp.zeros((3,))
c1 = jnp.zeros((3,))
A0 = jnp.zeros((3, 3))
A1 = jnp.zeros((3, 3))
B0 = jnp.zeros((3, 3))
B1 = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
b_sol = jnp.array([1, 0, 0])
gamma0 = jnp.array([1, 0, 0])
gamma1 = jnp.array([0, 0.5, -0.5])
b_error = None
register_stochastic_runge_kutta_method(
    "sri1", c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error
)

# SRI2
c0 = jnp.array([0, 1, 0.0])
c1 = jnp.array([0, 1, 1.0])
A0 = jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
A1 = jnp.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
B0 = jnp.zeros((3, 3))
B1 = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
b_sol = jnp.array([0.5, 0.5, 0])
gamma0 = jnp.array([1.0, 0, 0])
gamma1 = jnp.array([0, 0.5, -0.5])
b_error = None
register_stochastic_runge_kutta_method(
    "sri2", c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error
)


@partial(jax.jit, static_argnums=(1, 2, 5, 6, 7, 8, 9, 10,11))
def _sdeint_on_grid(
    key: PRNGKeyArray,
    drift: Callable,
    diffusion: Callable,
    y0: Array,
    ts: Array,
    step_fn: Callable,
    sde_type: str,
    noise_type: str,
    iterated_integral_fn: Union[Callable, None],
    return_brownian: bool = True,
    only_final: bool = False,
    scan_unroll: int = 1,
) -> Array:
    """Solve a stochastic differential equation on a grid.

    Args:
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        *args: Other arguments.

    Returns:
        Array: Solution of the SDE.
    """

    dts = ts[1:] - ts[:-1]

    def scan_fun(carry, data):
        key, y0, t0, f0, g0 = carry
        t1, dt = data
        sqrt_dt = jnp.sqrt(jnp.abs(dt))
        # Generate brownian increments
        key, key_dWt, key_dWtdWs = jrandom.split(key, 3)
        dWt = jrandom.normal(key_dWt, (noise_dim,)) * sqrt_dt
        if iterated_integral_fn is None:
            dWtdWs = None
        else:
            dWtdWs = iterated_integral_fn(key_dWtdWs, dWt, dt)

        y1, f1, g1, _ = step_fn(
            drift,
            diffusion,
            t0,
            y0,
            f0,
            g0,
            dt,
            dWt,
            dWtdWs,
            is_diagonal=is_diagonal_noise,
        )

        if return_brownian:
            if not only_final:
                return (key, y1, t1, f1, g1), (y1, dWt)
            else:
                return (key, y1, t1, f1, g1), None
        else:
            if not only_final:
                return (key, y1, t1, f1, g1), y1
            else:
                return (key, y1, t1, f1, g1), None

    t0 = ts[0]
    f0 = drift(t0, y0)
    g0 = diffusion(t0, y0)

    # Check if diffusion output is consistent with "noise_type"
    if g0.ndim < 2:
        noise_dim = g0.shape[0]
        is_diagonal_noise = noise_type == "diagonal"

    elif g0.ndim == 2:
        noise_dim = g0.shape[1]
        is_diagonal_noise = noise_type == "diagonal"
        assert (
            noise_type != "diagonal"
        ), "Noise type is set to be diagonal, but the diffusion function returns a matrix. Please set noise_type to 'general' or 'commutative' if your diffusion function returns a matrix."
    else:
        raise ValueError("Diffusion function must return a vector or matrix")

    init_carry = (key, y0, t0, f0, g0)
    if return_brownian:
        _, (ys, Ws) = lax.scan(scan_fun, init_carry, (ts[1:], dts), unroll=scan_unroll)
        return jnp.concatenate((y0[None], ys)), jnp.cumsum(
            jnp.concatenate((jnp.zeros_like(y0[None]), Ws), axis=0), axis=0
        )
    else:
        carry, ys = lax.scan(scan_fun, init_carry, (ts[1:], dts), unroll=scan_unroll)
        if only_final:
            return carry[1]
        else:
            return jnp.concatenate((y0[None], ys))

def sdeint(
    key: PRNGKeyArray,
    drift: Callable,
    diffusion: Callable,
    y0: Array,
    ts: Array,
    *args,
    method: str = "euler_maruyama",
    sde_type: str = "ito",
    noise_type: str = "general",
    return_brownian: bool = False,
    dt: Optional[Float] = None,
    rtol: Float = 1e-6,
    atol: Float = 1e-6,
    mxstep: Int = jnp.inf,
    dtmin: Float = 0.0,
    dtmax: Float = jnp.inf,
    scan_unroll: Int = 1,
) -> Array:
    """Solve a stochastic differential equation.

    Args:
        key: (PRNGKeyArray): Random generator key.
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        *args: Other arguments, that are passed both to the drift and diffusion functions i.e. parameters!
        method (str, optional): Methods to use. Defaults to "euler_maruyama".
        diagonal_noise (bool, optional): Whether the noise is diagonal. Defaults to False.
        dt (Optional[Float], optional): Fixed step size (optionally infered from ts). Defaults to None.
        rtol (Float, optional): Relative tolerance. Defaults to 1e-6.
        atol (Float, optional): Absolute tolerance. Defaults to 1e-6.
        mxstep (Int, optional): Maximum number of steps used by solver. Defaults to jnp.inf.
        dtmin (Float, optional): Minimal step size used by solver. Defaults to 0..
        dtmax (Float, optional): Maximal step size used by solver. Defaults to jnp.inf.

    Raises:
        TypeError: The arguments passed not jax types.
        TypeError: The arguments passed not jax types.

    Returns:
        ys: Solution path of the SDE.
    """

    y0 = jnp.atleast_1d(y0)
    ts = jnp.atleast_1d(ts)

    # Consistent dtype, based on the initial value.
    dtype = y0.dtype
    ts = ts.astype(dtype)

    # Make sure drift is consistent and is a function _f: R x R^d -> R^d where d >= 1
    # Make sure diffusion is consistent and is a function _g: R x R^d -> R^d (independent noise) where d >= 1 or _g: R x R^d -> R^{d x d} where d >= 1 (correlated noise)
    _f = lambda t, y: jnp.atleast_1d(drift(t, y, *args)).astype(dtype)
    if noise_type == "diagonal":
        _g = lambda t, y: jnp.atleast_1d(diffusion(t, y, *args)).astype(dtype)
    else:
        _g = lambda t, y: jnp.atleast_2d(diffusion(t, y, *args)).astype(dtype)

    # Get step_fn
    step_fn = get_step_fn(method, dtype=dtype)
    method_info = get_method_info(method)

    # Get necessary info
    requires_interated_integrals = method_info.get("requires_iterated_integrals", False)
    adaptive = method_info.get("adaptive", False)

    if requires_interated_integrals:
        iterated_integral_fn = get_iterated_integrals_fn(noise_type, sde_type)
    else:
        iterated_integral_fn = None

    return _sdeint_on_grid(
        key,
        _f,
        _g,
        y0,
        ts,
        step_fn,
        sde_type,
        noise_type,
        iterated_integral_fn=iterated_integral_fn,
        return_brownian=return_brownian,
        scan_unroll=scan_unroll,
    )
