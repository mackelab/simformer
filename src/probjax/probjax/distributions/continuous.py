import jax
import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf, gammaln, digamma

from jaxtyping import Array
from typing import Optional
from warnings import warn

from .exponential_family import ExponentialFamily
from .distribution import Distribution
from .constraints import (
    real,
    positive,
    strict_positive,
    strict_negative,
    unit_interval,
    simplex,
    square_matrix,
    strict_positive_integer,
    positive_definite_matrix,
    positive_integer,
    interval,
)
from .utils import _precision_to_scale_tril

from probjax.utils.linalg import batch_mv, batch_mahalanobis

__all__ = [
    "Normal",
    "MultivariateNormal",
    "Gamma",
    "Beta",
    "Uniform",
    "Cauchy",
    "Chi2",
    "Dirichlet",
    "Exp",
    "Laplace",
    "Logistic",
    "Pareto",
    "T",
    "TruncatedNormal",
    #    "GaussianKDE",
]

from jax.tree_util import register_pytree_node_class

# Implementations of distributions
from jax.scipy.stats import (
    norm,
    gamma,
    beta,
    expon,
    dirichlet,
    chi2,
    cauchy,
    gennorm,
    laplace,
    logistic,
    pareto,
    t,
    truncnorm,
    uniform,
    vonmises,
    gaussian_kde,
)


@register_pytree_node_class
class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Normal(jnp.array([0.0]), jnp.array([1.0]))
        >>> m.sample(key)  # normally distributed with loc=0 and scale=1
        array([-1.3348817], dtype=float32)

    Args:
        loc (float or ndarray): mean of the distribution (often referred to as mu)
        scale (float or ndarray): standard deviation of the distribution
            (often referred to as sigma)
    """

    arg_constraints = {"loc": real, "scale": strict_positive}
    support = real

    def __init__(self, loc: Array | float, scale: Array | float):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        self.loc, self.scale = jnp.broadcast_arrays(loc, scale)

        super().__init__(batch_shape=loc.shape, event_shape=())

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def median(self) -> Array:
        return self.loc

    @property
    def stddev(self) -> Array:
        return self.scale

    @property
    def variance(self) -> Array:
        return jnp.power(self.stddev, 2)

    @property
    def moment(self, n: int) -> Array:
        return self.scale * jnp.sqrt(2) * jnp.inverf(2 * n - 1)

    @property
    def fim(self) -> Array:
        mu = 1 / self.variance
        scale = 2 / self.variance
        mu_scale = jnp.stack([mu[..., None], scale[..., None]], axis=-1)
        return jnp.diag(mu_scale)

    def rsample(self, key, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape)
        return self.loc + eps * self.scale

    def log_prob(self, value) -> Array:
        return norm.logpdf(value, self.loc, self.scale)

    def cdf(self, value) -> Array:
        return norm.cdf(value, self.loc, self.scale)

    def icdf(self, value) -> Array:
        return norm.ppf(value, self.loc, self.scale)

    def entropy(self) -> Array:
        return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.scale)


@register_pytree_node_class
class MultivariateNormal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Normal(jnp.array([0.0]), jnp.array([1.0]))
        >>> m.sample(key)  # normally distributed with loc=0 and scale=1
        array([-1.3348817], dtype=float32)

    Args:
        loc (float or ndarray): mean of the distribution (often referred to as mu)
        covariance_matrix (float or ndarray): covariance matrix
        precision_matrix (float or ndarray): precision matrix
        scale_tril (float or ndarray): lower triangular matrix with positive
    """

    arg_constraints = {
        "loc": real,
        "cov": positive_definite_matrix,
        # "precision_matrix": square_matrix,
        # "scale_tril": square_matrix,
    }
    multivariate = True

    def __init__(
        self,
        loc: jnp.array,
        cov: Optional[jnp.array] = None,
        precision_matrix: Optional[jnp.array] = None,
        scale_tril: Optional[jnp.array] = None,
    ):
        if loc.ndim < 1:
            raise ValueError("loc must be at least one-dimensional.")

        if (cov is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.ndim < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                scale_tril.shape[:-2], loc.shape[:-1]
            )
            self.scale_tril = jnp.broadcast_to(
                scale_tril, batch_shape + scale_tril.shape[-2:]
            )
            self.cov = None
            self.precision_matrix = None
        elif cov is not None:
            if cov.ndim < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(cov.shape[:-2], loc.shape[:-1])

            self.cov = jnp.broadcast_to(cov, batch_shape + cov.shape[-2:])
            self.scale_tril = None
            self.precision_matrix = None
        else:
            if precision_matrix.ndim < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = jnp.broadcast_to(
                precision_matrix,
                batch_shape + precision_matrix.shape[-2:],
            )
            self.cov = None
            self.scale_tril = None

        self.loc = jnp.broadcast_to(loc, batch_shape + loc.shape[-1:])

        event_shape = self.loc.shape[-1:]
        batch_shape = batch_shape

        if cov is not None:
            self.scale_tril = jnp.linalg.cholesky(self.covariance_matrix)
        else:  # precision_matrix is not None
            self.scale_tril = _precision_to_scale_tril(self.precision_matrix)

        super().__init__(batch_shape, event_shape)

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def median(self) -> Array:
        return self.loc

    @property
    def variance(self) -> Array:
        return jnp.diagonal(self.covariance_matrix, axis1=-2, axis2=-1)

    @property
    def covariance_matrix(self) -> Array:
        if self.cov is not None:
            return self.cov
        else:
            return self.scale_tril @ self.scale_tril.T

    def rsample(self, key, sample_shape=()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape=shape, dtype=self.loc.dtype)
        return self.loc + batch_mv(self.scale_tril, eps)

    def log_prob(self, value: jnp.array) -> Array:

        diff = value - self.loc
        M = batch_mahalanobis(self.scale_tril, diff)
        half_log_det = jnp.sum(
            jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)),
            axis=-1,
        )
        return -0.5 * (self._event_shape[0] * jnp.log(2 * jnp.pi) + M) - half_log_det


@register_pytree_node_class
class Gamma(ExponentialFamily):
    r"""
    Creates a gamma distribution parameterized by shape `alpha` and rate `beta`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Gamma(jnp.array([2.0]), jnp.array([3.0]))
        >>> m.sample(key)  # gamma distribution with shape=2 and rate=3
        array([1.3750159], dtype=float32)

    Args:
        alpha (float or ndarray): shape parameter alpha
        beta (float or ndarray): rate parameter beta
    """

    arg_constraints = {"alpha": strict_positive, "beta": strict_positive}
    support = positive

    def __init__(self, alpha: Array, beta: Array):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha, self.beta = jnp.broadcast_arrays(alpha, beta)

        super().__init__(batch_shape=alpha.shape, event_shape=())

    @property
    def mean(self) -> Array:
        return self.alpha / self.beta

    @property
    def mode(self) -> Array:
        valid = self.alpha > 1
        return jnp.where(
            valid, (self.alpha - 1) / self.beta, jnp.zeros_like(self.alpha)
        )

    @property
    def variance(self) -> Array:
        return self.alpha / self.beta**2

    @property
    def moment(self, n: int) -> Array:
        assert all(n < self.beta), "n must be less than beta"
        return (1 - n / self.beta) ** (-self.alpha)

    @property
    def fim(self) -> Array:
        off_diag = -1 / self.beta
        diag11 = jnp.digamma(self.alpha)
        diag22 = self.alpha / self.beta**2

        fim = jnp.block(
            [
                [diag11[..., None], off_diag[..., None]],
                [off_diag[..., None], diag22[..., None]],
            ]
        )
        return fim

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.gamma(key, self.alpha, shape) / self.beta

    def log_prob(self, value):
        return gamma.logpdf(value * self.beta, self.alpha)

    def cdf(self, value):
        return gamma.cdf(value * self.beta, self.alpha)

    def icdf(self, value):
        return jax.scipy.special.gammaincinv(self.alpha, value) / self.beta

    def entropy(self):
        alpha, beta = self.alpha, self.beta
        return alpha - jnp.log(beta) + gammaln(alpha) + (1 - alpha) * digamma(alpha)


@register_pytree_node_class
class Beta(ExponentialFamily):
    r"""
    Creates a beta distribution parameterized by concentration parameters `alpha` and `beta`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Beta(jnp.array([2.0]), jnp.array([3.0]))
        >>> m.sample(key)  # beta distribution with alpha=2 and beta=3
        array([0.5302244], dtype=float32)

    Args:
        alpha (float or ndarray): concentration parameter alpha
        beta (float or ndarray): concentration parameter beta
    """

    arg_constraints = {"alpha": strict_positive, "beta": strict_positive}
    support = unit_interval

    def __init__(self, alpha: Array, beta: Array):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha, self.beta = jnp.broadcast_arrays(alpha, beta)

        super().__init__(batch_shape=alpha.shape, event_shape=())

    @property
    def concentration1(self) -> Array:
        return self.alpha

    @property
    def concentration0(self) -> Array:
        return self.beta

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.beta(key, self.alpha, self.beta, shape)

    def log_prob(self, value):
        # Numerical stability!
        value = jnp.clip(
            value, jnp.finfo(value.dtype).eps, 1.0 - jnp.finfo(value.dtype).eps
        )
        return beta.logpdf(value, self.alpha, self.beta)

    def cdf(self, value):
        return beta.cdf(value, self.alpha, self.beta)

    def entropy(self):
        alpha, beta = self.alpha, self.beta
        return (
            gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1) * digamma(alpha)
            + (beta - 1) * digamma(beta)
            - (alpha + beta - 2) * digamma(alpha + beta)
        )


@register_pytree_node_class
class Uniform(Distribution):
    arg_constraints = {"low": real, "high": real}

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

        self.low = jnp.where(low < high, low, high) - 1e-6

        super().__init__(batch_shape=jnp.shape(low), event_shape=())

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.uniform(key, shape, minval=self.low, maxval=self.high)

    def log_prob(self, value: Array) -> Array:
        return jnp.log(
            jnp.where(
                (value >= self.low) & (value <= self.high),
                1.0 / (self.high - self.low),
                0.0,
            )
        )

    def cdf(self, x: Array) -> Array:
        return jnp.where(
            x < self.low,
            0.0,
            jnp.where(x > self.high, 1.0, (x - self.low) / (self.high - self.low)),
        )

    def icdf(self, q: Array) -> Array:
        return self.low + q * (self.high - self.low)

    def entropy(self) -> Array:
        return jnp.log(self.high - self.low)


@register_pytree_node_class
class Cauchy(Distribution):
    arg_constraints = {"loc": real, "scale": strict_positive}

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

        super().__init__(batch_shape=jnp.shape(loc), event_shape=())

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def median(self) -> Array:
        return self.loc

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.cauchy(key, shape) * self.scale + self.loc

    def log_prob(self, value: Array) -> Array:
        return cauchy.logpdf(value, self.loc, self.scale)

    def cdf(self, x: Array) -> Array:
        return cauchy.cdf(x, self.loc, self.scale)

    def icdf(self, q: Array) -> Array:
        return cauchy.ppf(q, self.loc, self.scale)

    def entropy(self) -> Array:
        return jnp.log(4 * jnp.pi * self.scale)


@register_pytree_node_class
class Chi2(Distribution):
    arg_constraints = {
        "df": strict_positive_integer,
        "loc": real,
        "scale": strict_positive,
    }

    def __init__(self, df: float, loc: float, scale: float):
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)

        self.df = df
        self.loc = loc
        self.scale = scale

        super().__init__(batch_shape=jnp.shape(loc), event_shape=())

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.chisquare(key, self.df, shape) * self.scale + self.loc

    def log_prob(self, value: Array) -> Array:
        return chi2.logpdf(value, self.df, self.loc, self.scale)

    def cdf(self, x: Array) -> Array:
        return chi2.cdf(x, self.df, self.loc, self.scale)

    def icdf(self, q: Array) -> Array:
        return chi2.ppf(q, self.df, self.loc, self.scale)

    def entropy(self) -> Array:
        return jnp.log(0.5 * self.scale) + 0.5 * (1 + jnp.log(2 * jnp.pi * self.df))


@register_pytree_node_class
class Dirichlet(Distribution):
    arg_constraints = {"alpha": positive}
    multivariate = True

    def __init__(self, alpha: Array):
        assert jnp.ndim(alpha) >= 1, "alpha must be at least one-dimensional."
        assert jnp.all(alpha > 0), "alpha must be positive."

        self.alpha = alpha
        self.alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)

        if alpha.ndim > 1:
            batch_shape = jnp.shape(alpha)[:-1]
            event_shape = jnp.shape(alpha)[-1:]
        else:
            batch_shape = ()
            event_shape = jnp.shape(alpha)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def mean(self) -> Array:
        return self.alpha / self.alpha_sum

    @property
    def mode(self) -> Array:
        valid = self.alpha > 1
        return jnp.where(
            valid, (self.alpha - 1) / (self.alpha_sum - self.event_shape[0]), 0
        )

    @property
    def variance(self) -> Array:
        alpha_sum = self.alpha_sum
        return (
            self.alpha * (alpha_sum - self.alpha) / (alpha_sum**2 * (alpha_sum + 1))
        )

    @property
    def covariance_matrix(self) -> Array:
        alpha_sum = self.alpha_sum
        alpha_tilde = self.alpha / alpha_sum
        cov = jnp.diag(alpha_tilde) - alpha_tilde[..., None] * alpha_tilde[..., None, :]
        return cov / (alpha_sum[..., None, None] + 1)

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape  # Event shape is included in alpha
        return random.dirichlet(key, self.alpha, shape)

    def log_prob(self, value: Array) -> Array:
        alpha, value = jnp.broadcast_arrays(self.alpha, value)
        log_prob_fn = dirichlet.logpdf
        for _ in range(value.ndim - 1):
            log_prob_fn = jax.vmap(log_prob_fn)
        return log_prob_fn(value, alpha)

    def entropy(self) -> Array:
        return dirichlet.entropy(self.alpha)


@register_pytree_node_class
class Exp(ExponentialFamily):
    arg_constraints = {"rate": strict_positive}
    support = positive

    def __init__(self, rate: Array):
        self.rate = rate

        super().__init__(batch_shape=jnp.shape(rate), event_shape=())

    @property
    def mean(self) -> Array:
        return 1 / self.rate

    @property
    def variance(self) -> Array:
        return 1 / jnp.power(self.rate, 2)

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.exponential(key, shape) * 1 / self.rate

    def log_prob(self, value):
        return expon.logpdf(value, 0.0, 1 / self.rate)

    def cdf(self, value):
        return expon.cdf(value, 0.0, 1 / self.rate)

    def icdf(self, value):
        return expon.ppf(value, 0.0, 1 / self.rate)

    def entropy(self):
        return 1 - jnp.log(self.rate)


@register_pytree_node_class
class Laplace(ExponentialFamily):
    arg_constraints = {"loc": real, "scale": strict_positive}
    support = real

    def __init__(self, loc: Array, scale: Array):
        loc, scale = jnp.broadcast_arrays(loc, scale)

        self.loc = loc
        self.scale = scale

        super().__init__(batch_shape=jnp.shape(loc), event_shape=())

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def variance(self) -> Array:
        return 2 * jnp.power(self.scale, 2)

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.laplace(key, shape) * self.scale + self.loc

    def log_prob(self, value):
        return laplace.logpdf(value, self.loc, self.scale)

    def cdf(self, value):
        return laplace.cdf(value, self.loc, self.scale)

    def icdf(self, value):
        return laplace.ppf(value, self.loc, self.scale)

    def entropy(self):
        return 1 + jnp.log(2 * self.scale)


@register_pytree_node_class
class Logistic(ExponentialFamily):
    arg_constraints = {"loc": real, "scale": strict_positive}
    support = real

    def __init__(self, loc: Array, scale: Array):
        loc, scale = jnp.broadcast_arrays(loc, scale)

        self.loc = loc
        self.scale = scale

        super().__init__(batch_shape=jnp.shape(loc), event_shape=())

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def variance(self) -> Array:
        return jnp.power(self.scale * jnp.pi, 2) / 3

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def median(self) -> Array:
        return self.loc

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.logistic(key, shape) * self.scale + self.loc

    def log_prob(self, value):
        return logistic.logpdf(value, self.loc, self.scale)

    def cdf(self, value):
        return logistic.cdf(value, self.loc, self.scale)

    def icdf(self, value):
        return logistic.ppf(value, self.loc, self.scale)

    def entropy(self):
        return jnp.log(self.scale) + 2


@register_pytree_node_class
class Pareto(Distribution):
    arg_constraints = {"alpha": strict_positive, "scale": strict_positive}
    support = interval(1.0, jnp.inf)

    def __init__(self, alpha: Array, scale: Array):
        alpha, scale = jnp.broadcast_arrays(alpha, scale)

        self.alpha = alpha
        self.scale = scale

        super().__init__(batch_shape=jnp.shape(alpha), event_shape=())

    @property
    def mean(self) -> Array:
        return jnp.where(self.alpha > 1, self.scale / (self.alpha - 1), jnp.inf)

    @property
    def variance(self) -> Array:
        return jnp.where(
            self.alpha > 2,
            jnp.power(self.scale, 2) / ((self.alpha - 1) ** 2 * (self.alpha - 2)),
            jnp.inf,
        )

    @property
    def mode(self) -> Array:
        return self.scale

    @property
    def median(self) -> Array:
        return self.scale * 2 ** (1 / self.alpha)

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.pareto(key, self.alpha, shape) * self.scale

    def log_prob(self, value: Array) -> Array:
        return pareto.logpdf(value, self.alpha, 0.0, self.scale)

    def cdf(self, x: Array) -> Array:
        return pareto.cdf(x, self.alpha, 0.0, self.scale)

    def icdf(self, q: Array) -> Array:
        return pareto.ppf(q, self.alpha, 0.0, self.scale)

    def entropy(self) -> Array:
        return jnp.log(self.scale) + 1 + 1 / self.alpha


@register_pytree_node_class
class T(Distribution):
    arg_constraints = {
        "df": strict_positive_integer,
        "loc": real,
        "scale": strict_positive,
    }

    def __init__(self, df: Array, loc: Array, scale: Array):
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)

        self.df = df
        self.loc = loc
        self.scale = scale

        super().__init__(batch_shape=jnp.shape(df), event_shape=())

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.t(key, self.df, shape) * self.scale + self.loc

    def log_prob(self, value: Array) -> Array:
        return t.logpdf(value, self.df, self.loc, self.scale)

    def cdf(self, x: Array) -> Array:
        return t.cdf(x, self.df, self.loc, self.scale)

    def icdf(self, q: Array) -> Array:
        return t.ppf(q, self.df, self.loc, self.scale)

    def entropy(self) -> Array:
        return (
            jnp.log(self.scale)
            + 0.5 * (1 + jnp.log(self.df))
            + gammaln(0.5 * (self.df + 1))
            - gammaln(0.5 * self.df)
        )


@register_pytree_node_class
class TruncatedNormal(Distribution):
    arg_constraints = {"loc": real, "scale": strict_positive, "low": real, "high": real}

    def __init__(self, loc: Array, scale: Array, low: Array, high: Array):
        loc, scale, low, high = jnp.broadcast_arrays(loc, scale, low, high)

        self.loc = loc
        self.scale = scale
        self.low = jnp.minimum(low, high)
        self.high = jnp.maximum(low, high)

        super().__init__(batch_shape=jnp.shape(loc), event_shape=())

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return (
            random.truncated_normal(key, self.low, self.high, shape) * self.scale
            + self.loc
        )

    def log_prob(self, value: Array) -> Array:
        return truncnorm.logpdf(value, self.low, self.high, self.loc, self.scale)

    def cdf(self, x: Array) -> Array:
        return truncnorm.cdf(x, self.low, self.high, self.loc, self.scale)

    def icdf(self, q: Array) -> Array:
        return truncnorm.ppf(q, self.low, self.high, self.loc, self.scale)

    def entropy(self) -> Array:
        return (
            jnp.log(self.scale)
            + 0.5 * (1 + jnp.log(self.df))
            + gammaln(0.5 * (self.df + 1))
            - gammaln(0.5 * self.df)
        )


# @register_pytree_node_class
# class GaussianKDE(Distribution):
#     arg_constraints = {"values": real}

#     def __init__(self, values: Array):
#         self.data = values
#         self.kde = gaussian_kde(values)

#         super().__init__(batch_shape=(), event_shape=())

#     def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
#         shape = sample_shape + self.batch_shape + self.event_shape
#         return self.kde.resample(key, shape)

#     def log_prob(self, value: Array) -> Array:
#         return self.kde.logpdf(value)

#     def cdf(self, x: Array) -> Array:
#         return self.kde.cdf(x)

#     def icdf(self, q: Array) -> Array:
#         return self.kde.ppf(q)


# @register_pytree_node_class
# class VonMises(Distribution):
#     arg_constraints = {"loc": real, "concentration": positive}

#     def __init__(self, loc: Array, concentration: Array):
#         self.loc = loc
#         self.concentration = concentration

#         super().__init__(batch_shape=jnp.shape(loc), event_shape=())

#     def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
#         shape = sample_shape + self.batch_shape + self.event_shape
#         return random.vonmises(key, shape, self.loc, self.concentration)

#     def log_prob(self, value: Array) -> Array:
#         return vonmises.logpdf(value, self.loc, self.concentration)

#     def entropy(self) -> Array:
#         return vonmises.entropy(self.concentration)


# @register_pytree_node_class
# class Geometric(Distribution):
#     arg_constraints = {"p": unit_interval}

#     def __init__(self, p: Array):
#         self.p = p

#         super().__init__(batch_shape=jnp.shape(p), event_shape=())

#     def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
#         shape = sample_shape + self.batch_shape + self.event_shape
#         return random.geometric(key, self.p, shape)

#     def log_prob(self, value: Array) -> Array:
#         return geom.logpmf(value, self.p)

#     def entropy(self) -> Array:
#         return geom.entropy(self.p)
