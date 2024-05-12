import jax
import jax.numpy as jnp

from functools import partial


# General root-finding algorithm
def newton_raphson(f, x0, tol=1e-6, max_iter=50):
    """
    Newton-Raphson root-finding algorithm for a vector-valued function.

    Args:
    - f: A function that takes a vector x and returns a vector of the same shape.
    - x0: Initial guess for the root.
    - tol: Tolerance for stopping criterion (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).

    Returns:
    - x: The estimated root of the function.
    """
    x = x0
    shape = x.shape

    # Flatten
    def _f(x):
        y = f(x.reshape(shape))
        return y.reshape(-1)

    f_jax = jax.jacobian(_f)

    x = x.reshape(-1)

    def scan_fn(carry, i):
        tol_reached, x = carry

        def true_fn(x):
            return x, jnp.inf

        def false_fn(x):
            y = _f(x)
            J = f_jax(x)
            delta_x = jax.scipy.linalg.solve(J, -y)
            x = x + delta_x
            return x, jnp.linalg.norm(delta_x)

        x, delta_x = jax.lax.cond(tol_reached, true_fn, false_fn, x)
        tol_reached = delta_x < tol
        return (tol_reached, x), x

    tol_reached = False
    _, x = jax.lax.scan(scan_fn, (tol_reached, x), jnp.arange(max_iter))
    return x[-1].reshape(shape)


def root(fun, x0, args=(), method="newton-raphson", tol=1e-3, max_iter=100):
    """Find a root of a function, using a fixed point iteration.

    Args:
        fun (Callable): Function to find root of.
        x0 (Array): Initial value.
        args (tuple, optional): Extra arguments to pass to function. Defaults to ().
        method (str, optional): Method to use. Defaults to 'fixpoint'.
        tol (float, optional): Tolerance. Defaults to 1e-3.

    Returns:
        Array: Root of function.
    """

    # Dtype constraints on tolerance
    dtype = x0.dtype
    precission = jnp.finfo(dtype).precision
    tol = max(tol, precission)

    _f = lambda x: fun(x, *args)

    if method == "newton-raphson":
        return newton_raphson(_f, x0, tol=tol, max_iter=max_iter)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")


# Scalar root-finding algorithm


def root_scalar(
    fun, x0=None, args=(), bracket=None, method="bisection", tol=1e-3, max_iter=100
):
    # dtype = x0.dtype
    # precission = jnp.finfo(dtype).precision
    # tol = max(tol, precission)

    _f = lambda x: fun(x, *args)

    if bracket is None:
        # Automatically bracket the root
        raise NotImplementedError("Automatic bracketing not implemented.")

    if method == "bisection":
        assert bracket is not None, "Bracket must be provided for bisection method."
        return bisection_method(_f, bracket, tol=tol, max_iter=max_iter)


from functools import partial


@partial(jax.jit, static_argnums=(0, 2, 3))
def bisection_method(f, bracket, tol=1e-3, max_iter=100):
    a, b = bracket
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    def scan_fn(carry, i):
        tol_reached, a, b = carry

        def true_fn(a, b):
            return a, b, jnp.inf

        def false_fn(a, b):
            c = (a + b) / 2
            y = f(c)
            mask = jnp.sign(f(a)) * jnp.sign(y) < 0
            a = jnp.where(mask, a, c)
            b = jnp.where(mask, c, b)
            error = jnp.linalg.norm(y)
            return a, b, error

        a, b, error = jax.lax.cond(tol_reached, true_fn, false_fn, a, b)
        tol_reached = error < tol
        return (tol_reached, a, b), a

    tol_reached = True
    _, roots = jax.lax.scan(scan_fn, (tol_reached, a, b), jnp.arange(max_iter))

    return roots[-1]
