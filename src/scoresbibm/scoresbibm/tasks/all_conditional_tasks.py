from scoresbibm.tasks.base_task import AllConditionalTask
from scoresbibm.methods.models import AllConditionalReferenceModel
from scoresbibm.utils.condition_masks import get_condition_mask_fn

import jax
import jax.numpy as jnp

from functools import partial
import math

from probjax.core import joint_sample, log_potential_fn, rv
from probjax.inference.mcmc import MCMC
from probjax.inference.marcov_kernels import HMCKernel, GaussianMHKernel, SliceKernel
from probjax.distributions import Normal, Uniform, MultivariateNormal, Dirac
from probjax.utils.jaxutils import ravel_args



def nonlinear_gaussian_tree_task():
    """
    Nonlinear Gaussian Tree Task.

    This function defines a probabilistic model that represents a nonlinear Gaussian tree task.
    It generates random variables theta1, theta2, theta3, x1, x2, x3, and x4, and returns their names,
    a joint sampler, and a potential function.

    Returns:
        var_names (list): List of variable names.
        joint_sampler (callable): Joint sampler function.
        potential_fn (callable): Potential function.
    """

    def model(key):
        key1, key2, key3, key4, key5, key6, key7, key8 = jax.random.split(key, 8)
        theta1 = rv(Normal(jnp.zeros((1,)), 1.0), name="theta1")(key1)
        theta2 = rv(Normal(theta1, 1.0), name="theta2")(key2)
        theta3 = rv(Normal(theta1, 1.0), name="theta3")(key3)
        z1 = jnp.sin(theta2) ** 2
        z2 = 0.1 * theta2**2
        z3 = 0.1 * theta3**2
        z4 = jnp.cos(theta3) ** 2
        x1 = rv(Normal(z1, 0.2), name="x1")(key5)
        x2 = rv(Normal(z2, 0.2), name="x2")(key6)
        x3 = rv(Normal(z3, 0.6), name="x3")(key7)
        x4 = rv(Normal(z4, 0.1), name="x4")(key8)

    joint_sampler = joint_sample(model)
    potential_fn = log_potential_fn(joint_sampler)
    var_names = ["theta1", "theta2", "theta3", "x1", "x2", "x3", "x4"]

    return var_names, joint_sampler, potential_fn


def nonlinear_marcov_chain():
    """
    Generates a nonlinear Markov chain model.

    Returns:
        var_names (list): List of variable names.
        joint_sampler (function): Joint sampler function.
        potential_fn (function): Log potential function.
    """

    def model2(key):
        key_thetas, key_xs = jax.random.split(key, 2)

        key_thetas, key_theta0 = jax.random.split(key_thetas, 2)
        key_xs, key_x0 = jax.random.split(key_xs, 2)
        theta0 = rv(Normal(jnp.zeros((1,)), 0.5), name=f"theta0")(key_theta0)
        x0 = rv(Normal(theta0**2, 0.5), name=f"x0")(key_x0)
        for i in range(1, 10):
            key_thetas, key_theta_i = jax.random.split(key_thetas, 2)
            key_xs, key_xi = jax.random.split(key_xs, 2)
            theta_i = rv(Normal(theta0, 0.5), name=f"theta{i}")(key_theta_i)
            x1 = rv(Normal(theta_i**2, 0.5), name=f"x{i}")(key_xi)
            theta0 = theta_i

    var_names = [f"theta{i}" for i in range(10)] + [f"x{i}" for i in range(10)]
    joint_sampler = joint_sample(model2)
    potential_fn = log_potential_fn(joint_sampler)

    return var_names, joint_sampler, potential_fn


def slcp():
    """
    This function defines a probabilistic model for the SLCP task.

    Returns:
        var_names (list): List of variable names in the model.
        joint_sampler (callable): Function that samples from the joint distribution of the model.
        potential_fn (callable): Function that computes the log-potential of the model.
    """

    def model3(key):
        key1, key2, key3, key4, key5, key6, key7, key8, key9 = jax.random.split(key, 9)

        prior = Uniform(jnp.array([-3.0]), jnp.array([3.0]))
        theta0 = rv(prior, name="theta0")(key1)
        theta1 = rv(prior, name="theta1")(key2)
        theta2 = rv(prior, name="theta2")(key3)
        theta3 = rv(prior, name="theta3")(key4)
        theta4 = rv(prior, name="theta4")(key5)

        mean = jnp.stack([theta0, theta1]).squeeze()
        s_1 = theta2**2
        s_2 = theta3**2
        rho = jnp.tanh(theta4)
        eps = 0.000001
        cov = jnp.array(
            [[s_1**2, rho * s_1 * s_2], [rho * s_1 * s_2, s_2**2]]
        ).squeeze() + eps * jnp.eye(2)
        p = MultivariateNormal(mean, cov)
        x1 = rv(p, name="x0")(key6)
        x2 = rv(p, name="x1")(key7)
        x3 = rv(p, name="x2")(key8)
        x4 = rv(p, name="x3")(key9)

    var_names = [f"theta{i}" for i in range(5)] + [f"x{i}" for i in range(4)]
    joint_sampler = joint_sample(model3)
    potential_fn = log_potential_fn(joint_sampler)

    return var_names, joint_sampler, potential_fn


def two_moons():
    """
    Generates a simulator for the two moons problem.

    Returns:
        var_names (list): List of variable names.
        joint_sampler (function): Joint sampler function.
        potential_fn (function): Potential function.
    """
    sqrt2 = math.sqrt(2)

    def model5(key):
        """Simulator for the two moons problem."""
        key1, key2, key3, key4 = jax.random.split(key, 4)

        theta0 = rv(Uniform(jnp.array([-1.0]), jnp.array([1.0])), name="theta0")(key1)
        theta1 = rv(Uniform(jnp.array([-1.0]), jnp.array([1.0])), name="theta1")(key2)

        p = rv(Normal(jnp.array([0.1]), 0.01), "r")
        u = rv(Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi), "alpha")
        r = p(key3)
        alpha = u(key4)
        rhs1 = rv(
            Dirac((r * jnp.cos(alpha) + 0.25) + (-jnp.abs(theta0 + theta1) / sqrt2)),
            "x0",
        )(key3)
        rhs2 = rv(Dirac((r * jnp.sin(alpha)) + ((theta1 - theta0) / sqrt2)), "x1")(key4)

    def potential_fn(theta0, theta1, x0, x1):
        ang = -jnp.pi / 4.0
        c = jnp.cos(ang)
        s = jnp.sin(ang)
        z0 = -jnp.abs(c * theta0 - s * theta1)
        z1 = s * theta0 + c * theta1

        u = x0 - z0 - 0.25
        v = x1 - z1

        r = jnp.sqrt(u**2 + v**2)

        likelihood = jax.scipy.stats.norm.logpdf(r, 0.1, 0.01)
        likelihood = jnp.where(u < 0, -jnp.inf, likelihood)
        likelihood = jnp.where(
            (theta0 < -1.0) | (theta1 < -1.0) | (theta0 > 1.0) | (theta1 > 1.0),
            -jnp.inf,
            likelihood,
        )

        return likelihood

    var_names = [f"theta{i}" for i in range(2)] + [f"x{i}" for i in range(2)]
    joint_sampler = joint_sample(model5, rvs=var_names)  # Excluding latent variables

    return var_names, joint_sampler, potential_fn


class AllConditionalBMTask(AllConditionalTask):
    def __init__(self, name, builder, backend="jax") -> None:
        assert backend == "jax", "Only JAX backend is supported"
        super().__init__(name, backend)
        var_names, joint_sampler, potential_fn = builder()
        self.var_names = var_names
        self.var_sizes = jax.tree_util.tree_map(
            lambda x: x.shape[0], joint_sampler(jax.random.PRNGKey(0))
        )
        self.joint_sampler = joint_sampler
        self.potential_fn = potential_fn
        self.ravel_condition_mask = lambda x: x
        self.unravel_condition_mask = lambda x: x

    def get_joint_sampler(self):
        return self.joint_sampler

    def get_theta_dim(self):
        return sum([self.var_sizes[var] for var in self.var_names if "theta" in var])

    def get_x_dim(self):
        return sum([self.var_sizes[var] for var in self.var_names if "x" in var])
    
    def get_node_id(self):
        dim = self.get_theta_dim() + self.get_x_dim()
        return jnp.arange(dim)

    def get_observation_generator(self, condition_mask_fn="structured_random"):
        condition_mask_fn = get_condition_mask_fn(condition_mask_fn)
        def observation_generator(key):
            while True:
                key, key_sample, key_condition_mask = jax.random.split(key,3)
                condition_mask = condition_mask_fn(key_condition_mask, 1, self.get_theta_dim(), self.get_x_dim())[0]
                condition_mask = self.ravel_condition_mask(condition_mask)
                if jnp.all(condition_mask):
                    continue
                
                samples = self.joint_sampler(key_sample)
                conditioned_names = [
                    self.var_names[i]
                    for i in range(len(self.var_names))
                    if condition_mask[i]
                ]
                try:
                    x_o = jnp.concatenate(
                        [samples[var] for var in conditioned_names], axis=-1
                    )
                except:
                    x_o = jnp.array([])
                theta_o = jnp.concatenate(
                    [samples[var] for var in self.var_names if var not in conditioned_names], axis=-1
                )
                x_o = x_o.flatten()
                theta_o = theta_o.flatten()
                condition_mask = self.unravel_condition_mask(condition_mask)
                
                yield (condition_mask, x_o, theta_o)
                
        return observation_generator

    def get_data(self, num_samples: int, rng=None):
        keys = jax.random.split(rng, (num_samples,))
        samples = jax.vmap(self.joint_sampler)(keys)
        thetas = jnp.concatenate(
            [samples[var] for var in self.var_names if "theta" in var], axis=-1
        )
        xs = jnp.concatenate(
            [samples[var] for var in self.var_names if "x" in var], axis=-1
        )
        return {"theta":thetas, "x":xs}

    def _prepare_for_mcmc(self, key, condition_mask, x_o):
        condition_mask = self.ravel_condition_mask(condition_mask)
        conditioned_names = [
            self.var_names[i] for i in range(len(self.var_names)) if condition_mask[i]
        ]
        conditioned_sizes = [self.var_sizes[conditioned_names[0]]]
        for i in range(1, len(conditioned_names)):
            conditioned_sizes.append(
                conditioned_sizes[i - 1] + self.var_sizes[conditioned_names[i]]
            )
        *x_o, x_last = jnp.split(x_o, conditioned_sizes, axis=-1)
        assert x_last.shape[-1] == 0, "Last element of x_o should be empty"
        conditioned_nodes = {var: val for var, val in zip(conditioned_names, x_o)}

        init_vals = self.joint_sampler(key)

        for var in conditioned_nodes:
            del init_vals[var]

        init_vals_flat, unravel = ravel_args(init_vals)
        potential_fn = self.potential_fn

        @jax.jit
        def potential_fn_wrapper(vals):
            vals = unravel(vals)
            return potential_fn(**vals, **conditioned_nodes)

        return init_vals_flat, potential_fn_wrapper

    def _get_conditional_sample_fn(self):
        raise NotImplementedError

    def _get_joint_sample_fn(self):
        @jax.vmap
        def sample_fn(key, *args, **kwargs):
            samples = self.joint_sampler(key)
            return jnp.concatenate([samples[var] for var in self.var_names], axis=-1)

        return sample_fn

    def get_reference_sampler(self):
        conditional_sample_fn = self._get_conditional_sample_fn()
        joint_sample_fn = self._get_joint_sample_fn()

        def sample_fn_wrapper(num_samples, x_o, rng=None, condition_mask=None, **kwargs):
            rngs = jax.random.split(rng, (num_samples,))
            if jnp.any(condition_mask):
                samples = conditional_sample_fn(rngs, condition_mask, x_o)
            else:
                samples = joint_sample_fn(rngs)
            return samples

        model = AllConditionalReferenceModel(sample_fn_wrapper)
        model.set_default_node_id(self.var_names)
        return model


class TwoMoonsAllConditionalTask(AllConditionalBMTask):
    def __init__(self, backend="jax") -> None:
        super().__init__("two_moons_all_cond", two_moons, backend=backend)

    def get_base_mask_fn(self):
        thetas_mask = jnp.eye(2, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((2, 2), dtype=jnp.bool_))
        base_mask = jnp.block(
            [[thetas_mask, jnp.zeros((2, 2))], [jnp.ones((2, 2)), x_mask]]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn

    def _get_conditional_sample_fn(self):
        @partial(jax.vmap, in_axes=[0, None, None])
        def sample_fn(key, condition_mask, x_o):
            key_init, key_mcmc = jax.random.split(key, 2)
            init_vals_flat, potential_fn_wrapper = self._prepare_for_mcmc(
                key_init, condition_mask, x_o
            )

            kernel1 = SliceKernel()
            kernel2 = GaussianMHKernel(step_size=0.01)
            state = kernel1.init_state(key_mcmc, init_vals_flat)
            mcmc = MCMC(kernel1, potential_fn_wrapper)
            samples, state = mcmc.run(state, 1000)
            mcmc2 = MCMC(kernel2, potential_fn_wrapper)
            samples, state = mcmc2.run(state, 3000)

            return samples

        return sample_fn


class SLCPAllConditionalTask(AllConditionalBMTask):
    def __init__(self, backend="jax") -> None:
        super().__init__("slcp_all_cond", slcp, backend=backend)
        
        def ravel_condition_mask(condition_mask):
            thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond = jnp.split(condition_mask, [5,7,9,11], axis=-1)
            x1_cond = jnp.any(x1_cond, axis=-1)[None]
            x2_cond = jnp.any(x2_cond, axis=-1)[None]
            x3_cond = jnp.any(x3_cond, axis=-1)[None]
            x4_cond = jnp.any(x4_cond, axis=-1)[None]
            return jnp.hstack([thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond])
        def unravel_condition_mask(condition_mask):
            thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond = jnp.split(condition_mask, [5,6,7,8], axis=-1)
            x1_cond = jnp.repeat(x1_cond, 2, axis=-1)
            x2_cond = jnp.repeat(x2_cond, 2, axis=-1)
            x3_cond = jnp.repeat(x3_cond, 2, axis=-1)
            x4_cond = jnp.repeat(x4_cond, 2, axis=-1)
            return jnp.hstack([thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond])
        
        self.ravel_condition_mask = ravel_condition_mask
        self.unravel_condition_mask = unravel_condition_mask

    def get_base_mask_fn(self):
        theta_dim = 5
        x_dim = 8
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        # TODO This could be triangular -> DAG
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(
            *tuple([jnp.tril(jnp.ones((x_i_dim, x_i_dim), dtype=jnp.bool_))] * 4)
        )
        base_mask = jnp.block(
            [
                [thetas_mask, jnp.zeros((theta_dim, x_dim))],
                [jnp.ones((x_dim, theta_dim)), x_i_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            # If node_ids are permuted, we need to permute the base_mask
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn

    def _get_conditional_sample_fn(self):
        @partial(jax.vmap, in_axes=[0, None, None])
        def sample_fn(key, condition_mask, x_o):
            key_init, key_mcmc = jax.random.split(key, 2)
            init_vals_flat, potential_fn_wrapper = self._prepare_for_mcmc(
                key_init, condition_mask, x_o
            )

            kernel1 = SliceKernel()
            kernel2 = GaussianMHKernel(step_size=0.1)
            state = kernel1.init_state(key_mcmc, init_vals_flat)
            mcmc = MCMC(kernel1, potential_fn_wrapper)
            samples, state = mcmc.run(state, 600)
            mcmc2 = MCMC(kernel2, potential_fn_wrapper)
            samples, state = mcmc2.run(state, 2000)

            return samples

        return sample_fn


class NonlinearGaussianTreeAllConditionalTask(AllConditionalBMTask):
    def __init__(self, backend="jax") -> None:
        super().__init__("tree_all_cond", nonlinear_gaussian_tree_task, backend=backend)

    def _get_conditional_sample_fn(self):
        @partial(jax.vmap, in_axes=[0, None, None])
        def sample_fn(key, condition_mask, x_o):
            key_init, key_mcmc = jax.random.split(key, 2)
            init_vals_flat, potential_fn_wrapper = self._prepare_for_mcmc(
                key_init, condition_mask, x_o
            )

            kernel = HMCKernel()
            state = kernel.init_state(key_mcmc, init_vals_flat)
            mcmc = MCMC(kernel, potential_fn_wrapper)
            samples, state = mcmc.run(state, 5000)

            return samples

        return sample_fn

    def get_base_mask_fn(self):
        base_mask = jnp.array(
            [
                [True, False, False, False, False, False, False],
                [True, True, False, False, False, False, False],
                [True, False, True, False, False, False, False],
                [False, True, False, True, False, False, False],
                [False, True, False, False, True, False, False],
                [False, False, True, False, False, True, False],
                [False, False, True, False, False, False, True],
            ]
        )

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn


class NonlinearMarcovChainAllConditionalTask(AllConditionalBMTask):
    def __init__(self, backend="jax") -> None:
        super().__init__("marcov_chain_all_cond", nonlinear_marcov_chain, backend=backend)

    def get_base_mask_fn(self):
        # Marcovian structure
        theta_mask = jnp.eye(10, dtype=jnp.bool_) | jnp.eye(10, k=-1, dtype=jnp.bool_)
        xs_mask = jnp.eye(10, dtype=jnp.bool_)
        theta_xs_mask = jnp.eye(10, dtype=jnp.bool_)
        fill_mask = jnp.zeros((10, 10), dtype=jnp.bool_)
        base_mask = jnp.block([[theta_mask, fill_mask], [theta_xs_mask, xs_mask]])

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn

    def _get_conditional_sample_fn(self):
        @partial(jax.vmap, in_axes=[0, None, None])
        def sample_fn(key, condition_mask, x_o):
            key_init, key_mcmc = jax.random.split(key, 2)
            init_vals_flat, potential_fn_wrapper = self._prepare_for_mcmc(
                key_init, condition_mask, x_o
            )

            kernel = HMCKernel()
            state = kernel.init_state(key_mcmc, init_vals_flat)
            mcmc = MCMC(kernel, potential_fn_wrapper)
            samples, state = mcmc.run(state, 5000)

            return samples

        return sample_fn
