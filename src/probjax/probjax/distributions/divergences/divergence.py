from probjax.distributions.utils import Match
from probjax.distributions import Distribution
from probjax import distributions as dist

import jax
import warnings

__all__ = ["register_divergence", "divergence"]

_DIV_REGISTRY = {}
_DIV_MEMOIZE = {}


def register_divergence(name, type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(
            "Expected type_p to be a Distribution subclass but got {}".format(type_p)
        )
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError(
            "Expected type_q to be a Distribution subclass but got {}".format(type_q)
        )

    if name not in _DIV_REGISTRY:
        _DIV_REGISTRY[name] = {}
        _DIV_MEMOIZE[name] = {}

    def decorator(fun):
        _DIV_REGISTRY[name][type_p, type_q] = fun
        _DIV_MEMOIZE[name].clear()  # reset since lookup order may have changed
        return fun

    return decorator


def _dispatch(name, type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [
        (super_p, super_q)
        for super_p, super_q in _DIV_REGISTRY[name]
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    # mypy isn't smart enough to know that _Match implements __lt__
    # see: https://github.com/python/typing/issues/760#issuecomment-710670503
    left_p, left_q = min(Match(*m) for m in matches).types  # type: ignore[type-var]
    right_q, right_p = min(Match(*reversed(m)) for m in matches).types  # type: ignore[type-var]
    left_fun = _DIV_REGISTRY[name][left_p, left_q]
    right_fun = _DIV_REGISTRY[name][right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(
            "Ambiguous kl_divergence({}, {}). Please register_kl({}, {})".format(
                type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__
            ),
            RuntimeWarning,
        )
    return left_fun


def divergence(
    name: str, p: Distribution, q: Distribution, mc_samples=0, key=None, **kwargs
) -> jax.Array:
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.
        mc_samples (int): Number of samples to use for Monte Carlo approximation of KL divergence. Defaults to 0. Then only analytic expressions.
        key (jax.random.PRNGKey): Key for random number generation. Defaults to None. Only required if mc_samples > 0.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _DIV_MEMOIZE[name][type(p), type(q)]
    except KeyError:
        fun = _dispatch(name,type(p), type(q))
        _DIV_MEMOIZE[name][type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError(
            "No KL(p || q) is implemented for p type {} and q type {}".format(
                p.__class__.__name__, q.__class__.__name__
            )
        )
    return fun(p, q, mc_samples=mc_samples, key=key, **kwargs)
