import jax
import jax.numpy as jnp

from probjax.distributions.divergences.divergence import register_divergence, divergence
from probjax import distributions as dist

__all__ = ["kl_divergence"]

NAME = "kl"


def kl_divergence(p, q, mc_samples=0, key=None):
    return divergence(NAME, p, q, mc_samples=mc_samples, key=key)


@register_divergence(NAME, dist.Distribution, dist.Distribution)
def _kl_generic(p, q, mc_samples=0, key=None):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL divergence between distributions with different event shapes not supported"
        )

    assert (
        mc_samples >= 0
    ), "For general distirbutions we require mc_samples >= 0, to evaluate a Monte Carlo approximation of the KL divergence."
    assert key is not None, "Key must be provided if mc_samples > 0"

    if p.has_rsample:
        samples = p.rsample(key, (mc_samples,))
    else:
        samples = p.sample(key, (mc_samples,))
    log_prob_p = p.log_prob(samples)
    log_prob_q = q.log_prob(samples)
    return (log_prob_p - log_prob_q).mean(0)


@register_divergence(NAME, dist.Independent, dist.Independent)
def _kl_independent_independent(p, q, mc_samples=0, key=None):
    base_dist_p = p.base_dist
    base_dist_q = q.base_dist
    if len(base_dist_p) == len(base_dist_q):
        kl_base = jnp.stack(
            [
                kl_divergence(p, q, mc_samples=mc_samples, key=key)
                for p, q in zip(base_dist_p, base_dist_q)
            ],
            axis=-1,
        )
    else:
        raise ValueError(
            "KL divergence between distributions with different event shapes not supported"
        )
   
    return kl_base.sum(-1)


@register_divergence(NAME, dist.Independent, dist.Distribution)
def _kl_independent_generic(p, q, mc_samples=0, key=None):
    kl_base = kl_divergence(p.base_dist, q, mc_samples=mc_samples, key=key)
    return kl_base.sum(-1)


@register_divergence(NAME, dist.Bernoulli, dist.Bernoulli)
def _kl_bernoulli_bernoulli(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    probs_q = q.probs
    t1 = probs_p * jnp.log(probs_p / probs_q)
    t2 = (1 - probs_p) * jnp.log((1 - probs_p) / (1 - probs_q))
    return t1 + t2


@register_divergence(NAME, dist.Normal, dist.Normal)
def _kl_normal_normal(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log(scale_q / scale_p)
    t2 = (scale_p**2 + (loc_p - loc_q) ** 2) / (2 * scale_q**2) - 0.5
    return t1 + t2


@register_divergence(NAME, dist.MultivariateNormal, dist.MultivariateNormal)
def _kl_mvn_mvn(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale_tril
    loc_q, scale_q = q.loc, q.scale_tril
    t1 = jnp.linalg.slogdet(scale_p)[1] - jnp.linalg.slogdet(scale_q)[1]
    t2 = (
        jnp.trace(jnp.linalg.solve(scale_q, scale_p))
        + (loc_q - loc_p).T @ jnp.linalg.solve(scale_q, loc_q - loc_p)
        - p.event_shape[0]
    )
    return t1 + t2


@register_divergence(NAME, dist.Categorical, dist.Categorical)
def _kl_categorical_categorical(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    probs_q = q.probs
    return (probs_p * jnp.log(probs_p / probs_q)).sum(-1)


@register_divergence(NAME, dist.Dirichlet, dist.Dirichlet)
def _kl_dirichlet_dirichlet(p, q, mc_samples=0, key=None):
    alpha_p = p.concentration
    alpha_q = q.concentration
    t1 = jax.scipy.special.gammaln(alpha_p.sum(-1)) - jax.scipy.special.gammaln(
        alpha_q.sum(-1)
    )
    t2 = jax.scipy.special.gammaln(alpha_q).sum(-1) - jax.scipy.special.gammaln(
        alpha_p
    ).sum(-1)
    t3 = ((alpha_p - alpha_q) * (1 / alpha_q - 1 / alpha_p)).sum(-1)
    return t1 + t2 + t3


@register_divergence(NAME, dist.Gamma, dist.Gamma)
def _kl_gamma_gamma(p, q, mc_samples=0, key=None):
    alpha_p, beta_p = p.concentration, p.rate
    alpha_q, beta_q = q.concentration, q.rate
    t1 = jax.scipy.special.gammaln(alpha_p) - jax.scipy.special.gammaln(alpha_q)
    t2 = (alpha_p - alpha_q) * (jax.scipy.special.digamma(alpha_p) - beta_p / alpha_p)
    t3 = (beta_p - beta_q) * (alpha_p / beta_p - 1)
    return t1 + t2 + t3


@register_divergence(NAME, dist.Beta, dist.Beta)
def _kl_beta_beta(p, q, mc_samples=0, key=None):
    alpha_p, beta_p = p.concentration1, p.concentration0
    alpha_q, beta_q = q.concentration1, q.concentration0
    t1 = (
        jax.scipy.special.gammaln(alpha_p)
        + jax.scipy.special.gammaln(beta_p)
        - jax.scipy.special.gammaln(alpha_p + beta_p)
    )
    t2 = (
        jax.scipy.special.digamma(alpha_p) - jax.scipy.special.digamma(alpha_p + beta_p)
    ) * (alpha_p - alpha_q)
    t3 = (
        jax.scipy.special.digamma(beta_p) - jax.scipy.special.digamma(alpha_p + beta_p)
    ) * (beta_p - beta_q)
    return t1 + t2 + t3


@register_divergence(NAME, dist.Exp, dist.Exp)
def _kl_exponential_exponential(p, q, mc_samples=0, key=None):
    rate_p = p.rate
    rate_q = q.rate
    t1 = jnp.log(rate_p / rate_q)
    t2 = rate_p / rate_q - 1
    return t1 + t2


@register_divergence(NAME, dist.Laplace, dist.Laplace)
def _kl_laplace_laplace(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log(scale_p / scale_q)
    t2 = (scale_p / scale_q) + (loc_p - loc_q).abs() / scale_q - 1
    return t1 + t2


@register_divergence(NAME, dist.Poisson, dist.Poisson)
def _kl_poisson_poisson(p, q, mc_samples=0, key=None):
    rate_p = p.rate
    rate_q = q.rate
    t1 = rate_p * jnp.log(rate_p / rate_q)
    t2 = rate_p - rate_q
    return t1 + t2


@register_divergence(NAME, dist.Cauchy, dist.Cauchy)
def _kl_cauchy_cauchy(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log((scale_p + scale_q) ** 2 + (loc_p - loc_q) ** 2)
    t2 = jnp.log(4 * scale_p * scale_q)
    return t1 - t2


@register_divergence(NAME, dist.Pareto, dist.Pareto)
def _kl_pareto_pareto(p, q, mc_samples=0, key=None):
    scale_p, alpha_p = p.scale, p.alpha
    scale_q, alpha_q = q.scale, q.alpha
    scale_ratio = scale_p / scale_q
    alpha_ratio = alpha_p / alpha_q
    t1 = alpha_q * jnp.log(scale_ratio)
    t2 = -jnp.log(alpha_ratio)

    return t1 + t2 + alpha_ratio - 1


@register_divergence(NAME, dist.Bernoulli, dist.Poisson)
def _kl_bernoulli_poisson(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    rate_q = q.rate
    t1 = probs_p * jnp.log(probs_p / rate_q)
    t2 = (1 - probs_p) * jnp.log((1 - probs_p) / rate_q)
    return t1 + t2

@register_divergence(NAME, dist.Bernoulli, dist.Beta)
def _kl_bernoulli_beta(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    alpha_q, beta_q = q.concentration1, q.concentration0
    t1 = (alpha_q - 1) * jnp.log(probs_p)
    t2 = (beta_q - 1) * jnp.log(1 - probs_p)
    t3 = (
        jax.scipy.special.gammaln(alpha_q)
        + jax.scipy.special.gammaln(beta_q)
        - jax.scipy.special.gammaln(alpha_q + beta_q)
    )
    return t1 + t2 + t3

@register_divergence(NAME, dist.Bernoulli, dist.Gamma)
def _kl_bernoulli_gamma(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    alpha_q, beta_q = q.concentration, q.rate
    t1 = (alpha_q - 1) * jnp.log(probs_p)
    t2 = -beta_q * probs_p
    t3 = jax.scipy.special.gammaln(alpha_q) - alpha_q * jnp.log(beta_q)
    return t1 + t2 + t3

@register_divergence(NAME, dist.Bernoulli, dist.Dirichlet)
def _kl_bernoulli_dirichlet(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    alpha_q = q.concentration
    t1 = (alpha_q - 1) * jnp.log(probs_p)
    t2 = jax.scipy.special.gammaln(alpha_q.sum(-1)) - jax.scipy.special.gammaln(
        alpha_q
    ).sum(-1)
    return t1 + t2

@register_divergence(NAME, dist.Bernoulli, dist.Exp)
def _kl_bernoulli_exponential(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    rate_q = q.rate
    t1 = jnp.log(probs_p / rate_q)
    t2 = (1 - probs_p) / rate_q
    return t1 + t2

@register_divergence(NAME, dist.Bernoulli, dist.Laplace)
def _kl_bernoulli_laplace(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log(probs_p / (1 - probs_p))
    t2 = (1 - probs_p) * (loc_q - scale_q) / scale_q
    return t1 + t2

@register_divergence(NAME, dist.Bernoulli, dist.Cauchy)
def _kl_bernoulli_cauchy(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log(probs_p / (1 - probs_p))
    t2 = (1 - probs_p) * (loc_q - scale_q) / scale_q
    return t1 + t2

@register_divergence(NAME, dist.Bernoulli, dist.Pareto)
def _kl_bernoulli_pareto(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    scale_q, alpha_q = q.scale, q.alpha
    t1 = jnp.log(probs_p / (1 - probs_p))
    t2 = (1 - probs_p) * (alpha_q / (alpha_q - 1)) / scale_q
    return t1 + t2

