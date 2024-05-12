import pytest
import jax
import jax.numpy as jnp

import itertools

from probjax.distributions import continuous
from probjax.distributions import discrete
from probjax.distributions.independent import Independent
from probjax.distributions.mixture import Mixture
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.distributions import Distribution
from probjax.distributions.constraints import Constraint
from probjax.distributions.divergences.kl import kl_divergence, _kl_generic

from probjax.distributions.constraint_registry import transform_to


CONTINOUS_DIST = [getattr(continuous, name) for name in continuous.__all__]
DISCRETE_DIST = [getattr(discrete, name) for name in discrete.__all__]
SPECIAL_DIST = [Independent, Mixture, TransformedDistribution]


# Some helper functions
def sample_and_log_prob(p: Distribution, key: jax.random.PRNGKey, sample_shape):
    # Check log_prob and sample without sampling shape
    sample = p.sample(key, sample_shape)
    log_prob = p.log_prob(sample)

    assert (
        sample.shape == sample_shape + p.batch_shape + p.event_shape
    ), "Sample shape mismatch"
    assert log_prob.shape == sample_shape + p.batch_shape, "Log_prob shape mismatch"
    assert jnp.isfinite(log_prob).all(), "Log_prob is not finite for all samples"


def mean_and_var(p: Distribution, key: jax.random.PRNGKey):
    # Check mean and variance
    sample = p.sample(key, (10000,))
    mean = jnp.mean(sample, axis=0)
    var = jnp.var(sample, axis=0)
    std = jnp.sqrt(var)

    assert mean.shape == p.batch_shape + p.event_shape, "Mean shape mismatch"
    assert var.shape == p.batch_shape + p.event_shape, "Variance shape mismatch"

    try:
        # This can be infinite for some distributions
        true_mean = p.mean
        true_var = p.variance
        true_std = p.stddev

        mask = jnp.isfinite(true_mean)
        mean = jnp.where(mask, mean, true_mean)
        mask = jnp.isfinite(true_var)
        var = jnp.where(mask, var, true_var)
        mask = jnp.isfinite(true_std)
        std = jnp.where(mask, std, true_std)

        # Rather lose check as LLN may not hold
        assert jnp.allclose(
            true_mean, mean, atol=0.1, rtol=0.5
        ), "Mean is not close to sample mean"
        assert jnp.allclose(
            true_var, var, atol=0.1, rtol=0.5
        ), "Variance is not close to sample variance"
        assert jnp.allclose(
            true_std, std, atol=0.1, rtol=0.5
        ), "Standard deviation is not close to sample standard deviation"
    except AssertionError as e:
        raise e
    except NotImplementedError:
        pass


def cdf_icdf(p: Distribution, key: jax.random.PRNGKey):
    # Check cdf and icdf
    sample = p.sample(key, (10000,))
    eval_points = p.sample(key, (10,))

    empirical_cdf = jnp.mean(sample[:, None] <= eval_points[None, :], axis=0)

    try:
        cdf = p.cdf(eval_points)

        assert cdf.shape == eval_points.shape, "CDF shape mismatch"
        assert jnp.isfinite(cdf).all(), "CDF is not finite for all samples"

        assert jnp.allclose(
            empirical_cdf, cdf, atol=0.1, rtol=0.5
        ), "CDF is not close to empirical cdf"

        try:
            icdf = p.icdf(cdf)
            assert icdf.shape == eval_points.shape, "ICDF shape mismatch"
            assert jnp.isfinite(icdf).all(), "ICDF is not finite for all samples"
            assert jnp.allclose(
                eval_points, icdf, atol=0.1, rtol=0.5
            ), "ICDF is not close to sample"
        except NotImplementedError:
            pass
        except AttributeError:
            # If jax.scipy.stats is not available, this will be thrown
            pass
    except AssertionError as e:
        raise e
    except AttributeError:
        # If jax.scipy.stats is not available, this will be thrown
        pass
    except NotImplementedError:
        pass


def init_dist(dist: type[Distribution], key, shape=(1,)):
    event_shape = shape

    keys = jax.random.split(key, len(dist.arg_constraints))
    kwargs = dict(
        [
            (name, transform_to(constraint)(jax.random.normal(key, event_shape)))
            for (name, constraint), key in zip(dist.arg_constraints.items(), keys)
        ]
    )

    p = dist(**kwargs)
    print(kwargs, p)
    return p


def mode_correct(p: Distribution, key: jax.random.PRNGKey):
    # Check mode
    sample = p.sample(key, (10000,))
    log_prob_samples = p.log_prob(sample)
    mode = sample[jnp.argmax(log_prob_samples)]
    mode_log_prob = p.log_prob(mode)

    assert mode.shape == p.batch_shape + p.event_shape, "Mode shape mismatch"
    assert jnp.isfinite(mode).all(), "Mode is not finite"
    try:
        assert jnp.allclose(p.mode, mode, atol=0.5), "Mode is not close to sample mode"
        assert mode_log_prob <= p.log_prob(mode), "Mode log_prob is not maximum"
    except AssertionError as e:
        raise e
    except NotImplementedError:
        pass


@pytest.mark.parametrize("dist", CONTINOUS_DIST + DISCRETE_DIST + SPECIAL_DIST)
def test_distribution_class_attributes(dist: type[Distribution]):
    assert hasattr(dist, "arg_constraints") and isinstance(
        dist.arg_constraints, dict
    ), "Missing arg_constraints"
    assert hasattr(dist, "support") and isinstance(
        dist.support, Constraint
    ), "Missing support"
    assert hasattr(dist, "has_rsample") and isinstance(
        dist.has_rsample, bool
    ), "Missing has_rsample"


@pytest.mark.parametrize("dist", CONTINOUS_DIST + DISCRETE_DIST)
def test_base_distribution(dist: type[Distribution], shape=(1,), seed=0):
    # Initialize distributions

    key = jax.random.PRNGKey(seed)
    if dist.multivariate:
        shape = shape + (2,)
    p = init_dist(dist, key, shape)

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)
    cdf_icdf(p, key)

    # Check PyTree
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(
        p.sample(key, shape), q.sample(key, shape)
    ), "PyTree reconstruction mismatch"


@pytest.mark.parametrize("dist", CONTINOUS_DIST + DISCRETE_DIST)
def test_independent_distribution(dist: type[Distribution], shape=(2,), seed=0):
    # Initialize distributions

    key = jax.random.PRNGKey(seed)
    if dist.multivariate:
        shape = shape + (2,)
    p = init_dist(dist, key, shape)

    try:
        p = Independent(p, 1)
    except AssertionError:
        return

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)

    # Check PyTree
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(
        p.sample(key, shape), q.sample(key, shape)
    ), "PyTree reconstruction mismatch"


@pytest.mark.parametrize(
    "dist1, dist2", itertools.combinations(CONTINOUS_DIST + DISCRETE_DIST, 2)
)
def test_mixed_independent_distribution(
    dist1: type[Distribution], dist2: type[Distribution], shape=(1,), seed=0
):
    key = jax.random.PRNGKey(seed)

    p1 = init_dist(dist1, key, shape)
    p2 = init_dist(dist2, key, shape)
    # Batch shapes must be the same, which may not be true if we have multivariate and univarite dist!
    try:
        p = Independent([p1, p2], 1)
    except AssertionError:
        # If batch and event shapes are different, we can't make an independent distribution
        return

    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)


@pytest.mark.parametrize("dist", CONTINOUS_DIST)
def test_transformed_distribution(dist: type[Distribution], shape=(1,), seed=0):
    # Initialize distributions

    key = jax.random.PRNGKey(seed)
    p = init_dist(dist, key, shape=shape)

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    # This is not implemented
    # mean_and_var(p, key)
    # mode_correct(p, key)

    # Check PyTree
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(
        p.sample(key, shape), q.sample(key, shape)
    ), "PyTree reconstruction mismatch"


@pytest.mark.parametrize("dist", CONTINOUS_DIST + DISCRETE_DIST)
def test_mixture_distribution(dist: type[Distribution], shape=(1,), seed=0):
    key = jax.random.PRNGKey(seed)
    p1 = init_dist(dist, jax.random.PRNGKey(seed + 42), shape=shape)
    p2 = init_dist(dist, jax.random.PRNGKey(seed + 420000), shape=shape)

    p = Mixture(jnp.array([0.5, 0.5]), [p1, p2])

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)


@pytest.mark.parametrize(
    "dist1, dist2", list(itertools.combinations(CONTINOUS_DIST + DISCRETE_DIST, 2))
)
def test_kl_divergence(dist1, dist2, shape=(1,), seed=0):
    # Initialize distributions

    key1 = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed + 420000)
    p = init_dist(dist1, key1)
    q = init_dist(dist2, key2)

    # Check KL divergence
    try:
        dist = kl_divergence(p, q)
    except AssertionError:
        # This is intened to be thrown for distirbutions with no analytic KL divergence
        return
    except ValueError:
        # This is intened to be thrown for distirbutions with different event shapes
        return
    dist_mc = _kl_generic(p, q, mc_samples=10000, key=key1)

    assert dist.shape == p.batch_shape, "KL divergence shape mismatch"
    assert jnp.allclose(
        dist, dist_mc, atol=0.1, rtol=0.1
    ), "MC KL divergence is not close to analytic KL divergence"
