from probjax.distributions.discrete import Dirac
from scoresbibm.methods.models import AllConditionalReferenceModel
from scoresbibm.tasks.all_conditional_tasks import AllConditionalTask
import jax
import jax.numpy as jnp
import jax.random as jrandom

from probjax.utils.odeint import _odeint
from probjax.utils.jaxutils import ravel_args
from probjax.core import joint_sample, log_potential_fn,rv 
from probjax.distributions import Normal, Uniform, Independent, MultivariateNormal
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.inference.mcmc import MCMC 
from probjax.inference.marcov_kernels import HMCKernel, GaussianMHKernel, SliceKernel, LangevianMHKernel

from functools import partial

from scoresbibm.utils.condition_masks import get_condition_mask_fn

def drift_lotka_volterra(t, data, alpha,beta, gamma, delta):
    predator, prey = data
    # predator and prey are swapped
    d_predator = alpha * predator - beta * predator * prey
    d_prey = -gamma * prey + delta * predator * prey
    return d_predator, d_prey


def rbf_kernel(x1, x2, l = 7., sigma_f=2.5):
    dists = (x1[None, :]- x2[:, None])**2
    return sigma_f**2 * jnp.exp(-0.5 * dists / l**2) + 1e-5 * jnp.eye(x1.shape[0])

def betas_gp(key, t, kernel_fn=rbf_kernel):
    K = kernel_fn(t,t)
    p0 = MultivariateNormal(jnp.zeros_like(t), K)
    p = TransformedDistribution(p0, lambda x: jax.nn.sigmoid(x))
    return rv(p, name="theta2")(key)

def beta_fn(t,ts, betas):
    y = jnp.interp(t, ts, betas)
    return y

def sir_model(t, y, gamma, delta, betas, ts_betas):
    S, I, R,D = y
    dSdt = -beta_fn(t, ts_betas,betas) * S * I
    dIdt = beta_fn(t, ts_betas,betas) * S * I - (gamma + delta) * I
    dRdt = gamma * I
    dDdt = delta * I
    return dSdt, dIdt, dRdt,dDdt

def lotka_volterra(time_start = 0,time_end = 15, eval_time_points=150, num_timepoints=30):
    
    
        def subsample_data(key, data_batch, node_id, meta_data):
            num_devices = data_batch.shape[0]
            batch_size = data_batch.shape[1]
            data_batch = data_batch.reshape((num_devices * batch_size, -1, 1))
            n = num_devices * batch_size
            key_t1, key_t2 = jrandom.split(key, 2)
            ts_dense = jnp.linspace(time_start, time_end, eval_time_points) 
            # Random time points 
            ts1 = jrandom.uniform(key_t1, (n , num_timepoints), minval=time_start, maxval=time_end)
            ts2 = jrandom.uniform(key_t2, (n , num_timepoints), minval=time_start, maxval=time_end)
            
            ts1 = jnp.sort(ts1, axis=1)
            ts2 = jnp.sort(ts2, axis=1)
            
            predator = data_batch[:, 4:4+eval_time_points,0]
            prey = data_batch[:, 4+eval_time_points:,0]
            predator = jax.vmap(lambda *args: jnp.interp(*args), in_axes=(0, None, 0))(ts1, ts_dense, predator)
            prey = jax.vmap(lambda *args: jnp.interp(*args), in_axes=(0, None, 0))(ts2, ts_dense, prey)

            node_ids = jnp.array([0, 1, 2, 3] + [4] * num_timepoints + [5] * num_timepoints)
            node_ids = jnp.repeat(node_ids[None, ...], num_devices, axis=0)
            node_metadata = jnp.concatenate([jnp.full((n, 4), jnp.nan),ts1, ts2], axis=1).reshape((num_devices, batch_size, -1, 1))
            full_data = jnp.concatenate([data_batch[:, :4,0], predator, prey], axis=1).reshape((num_devices, batch_size, -1, 1))
            return full_data, node_ids, node_metadata


        def dense_meta_data():
            ts_dense = jnp.linspace(time_start, time_end, eval_time_points)
            meta_data = { "theta0": jnp.array([jnp.nan]),"theta1": jnp.array([jnp.nan]),"theta2": jnp.array([jnp.nan]),"theta3": jnp.array([jnp.nan]), "x0": ts_dense, "x1": ts_dense}
            return meta_data
            

        def model(key, ts1, ts2, ode_method="rk4", ode_ts_grid=jnp.linspace(time_start, time_end, eval_time_points)):
            key_theta0, key_theta1, key_theta2, key_theta3, key_predator, key_prey = jrandom.split(key, 6)
            prior = TransformedDistribution(Normal(jnp.zeros(1), jnp.ones(1)), lambda x: jax.nn.sigmoid(x)*2 + 1.)
            theta0 = rv(prior, name="theta0")(key_theta0)
            theta1 = rv(prior, name="theta1")(key_theta1)
            theta2 = rv(prior, name="theta2")(key_theta2)
            theta3 = rv(prior, name="theta3")(key_theta3)
            
            predator, prey = _odeint(drift_lotka_volterra, (1., 0.5), ode_ts_grid, theta0, theta1, theta2, theta3, method=ode_method)
            
            predator_observed_mean = jnp.interp(ts1, ode_ts_grid, predator)
            prey_observed_mean = jnp.interp(ts2, ode_ts_grid, prey)
            
            x0 = rv(Independent(Normal(predator_observed_mean, 0.1),1), name="x0")(key_predator)
            x1 = rv(Independent(Normal(prey_observed_mean, 0.1),1), name="x1")(key_prey)
            
            # x0_dense = rv(Dirac(predator), name="x0_dense")(key_predator)
            # x1_dense = rv(Dirac(prey), name="x1_dense")(key_prey)
        
        
        var_names = ["theta0", "theta1", "theta2", "theta3", "x0", "x1"]
        
        return model, dense_meta_data, subsample_data, var_names
        

def sir(time_start = 0,time_end = 50, eval_time_points=100, num_timepoints=30):
    
    varnames = ["theta0", "theta1", "theta2", "x0", "x1", "x2"]
    
    def model(key, ts1, ts2, ts3, ts4, ode_method="rk4", ode_ts_grid=jnp.linspace(time_start, time_end, eval_time_points)):
        key_theta0, key_theta1, key_theta2 , key_I, key_D, key_R = jrandom.split(key, 6)
        prior_gamma_delta = Uniform(jnp.zeros(1), jnp.array([0.5]))
        theta0 = rv(prior_gamma_delta, name="theta0")(key_theta0)
        theta1 = rv(prior_gamma_delta, name="theta1")(key_theta1)
        theta3 = betas_gp(key_theta2, ts1)
        
        S0 = 1.
        R0 = 0.
        D0 = 0.
        I0 = jax.random.uniform(key_I, (), minval=0., maxval=0.2)
        S, I, R, D = _odeint(sir_model, (S0, I0, R0,D0), ode_ts_grid, theta0, theta1, theta3, ts1, method=ode_method)
        
        
        I = jnp.interp(ts2, ode_ts_grid, I)
        R = jnp.interp(ts3, ode_ts_grid, R)
        D = jnp.interp(ts4, ode_ts_grid, D)
        
        I_log = jnp.log(I + 1e-8)
        R_log = jnp.log(R + 1e-8)
        D_log = jnp.log(D + 1e-8)
        
        p_observed1 = TransformedDistribution(Independent(Normal(I_log, 0.05), 1), lambda x: jnp.exp(x))
        p_observed2 = TransformedDistribution(Independent(Normal(R_log, 0.05), 1), lambda x: jnp.exp(x))
        p_observed3 = TransformedDistribution(Independent(Normal(D_log, 0.05), 1), lambda x: jnp.exp(x))
        I_obs = rv(p_observed1, name="x0")(key_I)
        R_obs = rv(p_observed2, name="x1")(key_D)
        D_obs = rv(p_observed3, name="x2")(key_R)
        
        
    def dense_meta_data():
        ts_dense = jnp.linspace(time_start, time_end, eval_time_points)
        meta_data = { "theta0": jnp.array([jnp.nan]),"theta1": jnp.array([jnp.nan]),"theta2": ts_dense, "x0": ts_dense, "x1": ts_dense, "x2": ts_dense}
        return meta_data
    
    def subsample_data(key, data_batch, node_id, meta_data):
        num_devices = data_batch.shape[0]
        batch_size = data_batch.shape[1]
        data_batch = data_batch.reshape((num_devices * batch_size, -1, 1))
        n = num_devices * batch_size
        key_t1, key_t2, key_t3, key_t4 = jrandom.split(key, 4)
        ts_dense = jnp.linspace(time_start, time_end, eval_time_points) 
        # Random time points 
        ts1 = jrandom.uniform(key_t1, (n , num_timepoints), minval=time_start, maxval=time_end)
        ts2 = jrandom.uniform(key_t2, (n , num_timepoints), minval=time_start, maxval=time_end)
        ts3 = jrandom.uniform(key_t3, (n , num_timepoints), minval=time_start, maxval=time_end)
        ts4 = jrandom.uniform(key_t4, (n , num_timepoints), minval=time_start, maxval=time_end)
        
        ts1 = jnp.sort(ts1, axis=1)
        ts2 = jnp.sort(ts2, axis=1)
        ts3 = jnp.sort(ts3, axis=1)
        ts4 = jnp.sort(ts4, axis=1)
        
        betas = data_batch[:, 2:2+eval_time_points,0]
        I = data_batch[:, 2+eval_time_points: 2 + 2*eval_time_points,0]
        R = data_batch[:, 2+eval_time_points*2: 2 + eval_time_points*3,0]
        D = data_batch[:, 2+eval_time_points*3: 2 + eval_time_points*4,0]
        
        betas = jax.vmap(lambda *args: jnp.interp(*args), in_axes=(0, None, 0))(ts1, ts_dense, betas)
        I = jax.vmap(lambda *args: jnp.interp(*args), in_axes=(0, None, 0))(ts2, ts_dense, I)
        R = jax.vmap(lambda *args: jnp.interp(*args), in_axes=(0, None, 0))(ts3, ts_dense, R)
        D = jax.vmap(lambda *args: jnp.interp(*args), in_axes=(0, None, 0))(ts4, ts_dense, D)

        node_ids = jnp.array([0, 1] + [2] * num_timepoints + [3] * num_timepoints + [4] * num_timepoints + [5] * num_timepoints)
        node_ids = jnp.repeat(node_ids[None, ...], num_devices, axis=0)
        node_metadata = jnp.concatenate([jnp.full((batch_size, 2), jnp.nan),ts1, ts2, ts3, ts4], axis=1).reshape((num_devices,batch_size, -1, 1))
        full_data = jnp.concatenate([data_batch[:, :2, 0], betas, I, R, D], axis=1).reshape((num_devices,batch_size, -1, 1))
        return full_data, node_ids, node_metadata
    
    return model, dense_meta_data, subsample_data, varnames 
        
        
    

class UnstructuredTask(AllConditionalTask):
    
    def __init__(self, name: str, builder, backend: str = "jax", **kwargs) -> None:
        model, meta_data, subsampler, var_names = builder(**kwargs)
        self.var_names = var_names
        self.model = model
        self.subsampler = subsampler
        self.dense_meta_data = meta_data()
        self.joint_sampler = joint_sample(model, rvs=self.var_names)
        
        self.ravel_condition_mask = lambda x: x
        self.unravel_condition_mask = lambda x: x
        
        self.ravel_meta_data = lambda x: x
        self.unravel_meta_data = lambda x: x
        
        super().__init__(name, backend)
        
    def sample_meta_data(self, key):
        raise NotImplementedError()
    
    def gat_eval_node_id(self):
        raise NotImplementedError()
        
    def get_observation_generator(self, condition_mask_fn="structured_random"):
        condition_mask_fn = get_condition_mask_fn(condition_mask_fn)
        def observation_generator(key):
            while True:
                key, key_sample, key_condition_mask, key_meta_data = jax.random.split(key,4)
                condition_mask = condition_mask_fn(key_condition_mask, 1, self.get_theta_dim(), self.get_x_dim())[0]
                condition_mask = self.ravel_condition_mask(condition_mask)
                condition_mask = jax.lax.cond(jnp.all(condition_mask), lambda x: jnp.zeros_like(x), lambda x: x, condition_mask)
                meta_data = self.sample_meta_data(key_meta_data)
                
                samples = self.joint_sampler(key_sample, *meta_data)
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
                meta_data = self.ravel_meta_data(*meta_data)
                node_id = self.get_eval_node_id()
                
                
                yield (condition_mask, x_o, theta_o, meta_data, node_id)
                
        return observation_generator
        
        
    def get_data(self, num_samples: int, rng=None):
        rngs = jax.random.split(rng, (num_samples,))
        required_meta_data = [self.dense_meta_data[var] for var in self.var_names if not jnp.isnan(self.dense_meta_data[var]).any()]
        samples = jax.vmap(self.joint_sampler, in_axes=(0,) + (None,)*len(required_meta_data))(rngs, *required_meta_data)
        thetas = jnp.concatenate([samples[var] for var in self.var_names if var.startswith("theta")], axis=-1)
        xs = jnp.concatenate([samples[var] for var in self.var_names if var.startswith("x")], axis=-1)
        dense_meta_data = jnp.concatenate([self.dense_meta_data[var] for var in self.var_names], axis=-1)
        data = {"theta": thetas, "x": xs, "metadata":dense_meta_data}

        return data
    
    def get_base_mask_fn(self):
        
        return lambda *args, **kwargs: None   
    
    def get_batch_sampler(self):
        base_batch_sampler = super().get_batch_sampler()
        
        @partial(jax.jit, static_argnums=(1, 5))
        def batch_sampler(key, batch_size, data, node_id, meta_data=None, num_devices=1):
            data, node_id, meta_data = base_batch_sampler(key, batch_size, data, node_id, meta_data, num_devices)
            return self.subsampler(key, data, node_id, meta_data)
        return batch_sampler
    
        
    
    def _prepare_for_mcmc(self, key, condition_mask, x_o, meta_data):
        condition_mask = self.unravel_condition_mask(condition_mask)
        meta_data = self.unravel_meta_data(meta_data)

    
        init_vals = self.joint_sampler(key, *meta_data)


        conditioned_names = [self.var_names[i] for i in range(len(self.var_names)) if condition_mask[i]]
        varsizes = jnp.array([init_vals[var].shape[-1] for var in conditioned_names])
        x_o_split = jnp.split(x_o, jnp.cumsum(varsizes)[:-1], axis=-1)
        
        conditioned_nodes = {var: x_o_split[i] for i, var in enumerate(conditioned_names)}

        for var in conditioned_nodes:
            del init_vals[var]

        init_vals_flat, unravel = ravel_args(init_vals)
        potential_fn = log_potential_fn(self.model, *meta_data)
        
        @jax.jit
        def potential_fn_wrapper(vals):
            vals = unravel(vals)
            return potential_fn(**vals, **conditioned_nodes) 

        return init_vals_flat, potential_fn_wrapper
        
    def _get_conditional_sample_fn(self):
        raise NotImplementedError

    def _get_joint_sample_fn(self):
        raise NotImplementedError

    def get_reference_sampler(self):
        conditional_sample_fn = self._get_conditional_sample_fn()
        joint_sample_fn = self._get_joint_sample_fn()

        def sample_fn_wrapper(num_samples, x_o, rng=None, condition_mask=None, meta_data=None, **kwargs):
            rngs = jax.random.split(rng, (num_samples,))
            if jnp.any(condition_mask):
                samples = conditional_sample_fn(rngs, condition_mask, x_o, meta_data)
            else:
                samples = joint_sample_fn(rngs, meta_data)
            return samples

        model = AllConditionalReferenceModel(sample_fn_wrapper)
        model.set_default_node_id(self.var_names)
        return model
        
    
class LotkaVolterraTask(UnstructuredTask):
    
    def __init__(self, time_start = 0,time_end = 15, eval_time_points=150, num_timepoints=20, backend: str = "jax") -> None:
        self.time_start = time_start
        self.time_end = time_end
        self.eval_time_points = eval_time_points
        self.num_timepoints = num_timepoints
        super().__init__("lotka_volterra", lotka_volterra,backend= backend, time_start = time_start,time_end = time_end, eval_time_points=eval_time_points, num_timepoints=num_timepoints)
        
        def ravel_meta_data(*meta_data):
            return jnp.concatenate((jnp.full((4,), jnp.nan), *meta_data))
        
        def unravel_meta_data(meta_data):
            return jnp.split(meta_data[4:], 2, axis=-1)
        
        def ravel_condition_mask(condition_mask):
            mask_theta, mask_x0, mask_x1 = jnp.split(condition_mask, [4, 4 + self.num_timepoints], axis=-1)
            mask_x0 = jnp.any(mask_x0)[None]
            mask_x1 = jnp.any(mask_x1)[None]
            return jnp.concatenate((mask_theta, mask_x0, mask_x1))
        
        def unravel_condition_mask(condition_mask):
            mask_theta, mask_x0, mask_x1 = jnp.split(condition_mask, [4, 5], axis=-1)
            mask_x0 = jnp.repeat(mask_x0, self.num_timepoints, axis=-1)
            mask_x1 = jnp.repeat(mask_x1, self.num_timepoints, axis=-1)
            return jnp.concatenate((mask_theta, mask_x0, mask_x1))
            
        self.ravel_meta_data = ravel_meta_data
        self.unravel_meta_data = unravel_meta_data
        self.ravel_condition_mask = ravel_condition_mask
        self.unravel_condition_mask = unravel_condition_mask
        
    def get_node_id(self):
        num_timepoints = self.dense_meta_data["x0"].shape[0]
        return jnp.array([0, 1, 2, 3] + [4] * num_timepoints + [5] * num_timepoints)
    
    def get_eval_node_id(self):
        return jnp.array([0, 1, 2, 3] + [4] * self.num_timepoints + [5] * self.num_timepoints)
    
    def get_x_dim(self):
        return 2* self.num_timepoints
    
    def get_theta_dim(self):
        return 4
    
    def sample_meta_data(self, key):
        key1, key2 = jrandom.split(key, 2)
        ts1 = jrandom.uniform(key1, (self.num_timepoints,), minval=self.time_start, maxval=self.time_end)
        ts2 = jrandom.uniform(key2, (self.num_timepoints,), minval=self.time_start, maxval=self.time_end)
        ts1 = jnp.sort(ts1)
        ts2 = jnp.sort(ts2)
        return (ts1, ts2)    
    
    def _get_conditional_sample_fn(self):
        
        @partial(jax.vmap, in_axes = [0, None, None, None])
        def sample_fn(key, condition_mask, x_o, meta_data, **kwargs):
            key_init, key_sample = jax.random.split(key, 2)
            init_vals_flat, potential_fn_wrapper = self._prepare_for_mcmc(key_init, condition_mask, x_o, meta_data)

            
            
            kernel = GaussianMHKernel(step_size=0.5)
            kernel2 = SliceKernel(step_size=0.01)
            state = kernel.init_state(key_sample,init_vals_flat)
            mcmc = MCMC(kernel, potential_fn_wrapper)
            mcmc2 = MCMC(kernel2, potential_fn_wrapper)
            samples, state = mcmc.run(state, 10000)
            samples, state = mcmc2.run(state, 500)
            return samples
        
        return sample_fn
    
    def _get_joint_sample_fn(self):
        @partial(jax.vmap, in_axes = [0, None])
        def sample_fn(key, meta_data, **kwargs):
            ts1, ts2 = self.unravel_meta_data(meta_data)
            samples = self.joint_sampler(key, ts1, ts2)
            return jnp.concatenate([samples[var] for var in self.var_names], axis=-1)

        return sample_fn
    
    def get_base_mask_fn(self, meta_data_dependent=True):
        
        def base_mask_fn(node_id, meta_data):
            meta_data = meta_data.reshape(node_id.shape)
            id_params1 = (node_id == 0) | (node_id == 1)
            id_params2 = (node_id == 2) | (node_id == 3)
            id_mask1 = node_id == 4
            id_mask2 = node_id == 5

            max_points = node_id.shape[0]
            indices1 = jnp.where(id_mask1, size=max_points)[0]
            indices2 = jnp.where(id_mask2, size=max_points)[0]

            mask = jnp.eye(node_id.shape[0], dtype=bool)
            if meta_data_dependent:
                in_past = meta_data[None,:] < meta_data[:, None]
                in_past2 = meta_data[None,:] < meta_data[:, None]
                # distances = jnp.abs(meta_data[None,:] - meta_data[:, None])
                # distances = distances / in_past
                # cross_distances = distances * (id_mask1[None, :] & id_mask2[:, None]) + ~(id_mask1[None, :] & id_mask2[:, None]) * jnp.inf
                # corss_distances2 = distances * (id_mask2[None, :] & id_mask1[:, None]) + ~(id_mask2[None, :] & id_mask1[:, None]) * jnp.inf
                # closest_value = jnp.argsort(cross_distances, axis=1)
                # closest_value2 = jnp.argsort(corss_distances2, axis=1)
                # mask = mask.at[jnp.arange(mask.shape[0]), closest_value[:,0]].set(True)
                # mask = mask.at[jnp.arange(mask.shape[0]),closest_value2[:,0]].set(True)
                # mask = mask.at[jnp.arange(mask.shape[0]), closest_value[:,1]].set(True)
                # mask = mask.at[jnp.arange(mask.shape[0]),closest_value2[:,1]].set(True)
                # mask = mask.at[1, :].set(False)
                # mask = mask.at[:,1].set(False)
                # mask = mask.at[1, 1].set(True)
                in_past1 = in_past & (id_mask1[None, :] & id_mask2[:, None])
                in_past2 = in_past2 & (id_mask2[None, :] & id_mask1[:, None])
                mask = mask | in_past1
                mask = mask | in_past2
            else:
                mask = mask | (id_mask1[None, :] & id_mask2[:, None])
                #mask = mask | (id_mask2[None, :] & id_mask1[:, None])

            mask = mask.at[indices1[1:], indices1[:-1]].set(True)
            mask = mask.at[indices2[1:], indices2[:-1]].set(True)
            mask = mask.at[0, :].set(False)
            mask = mask.at[:, 0].set(False)
            mask = mask.at[0, 0].set(True)
            mask = mask | id_params1[None,:] & id_mask1[:, None]
            mask = mask | id_params2[None,:] & id_mask2[:, None]
            return mask.astype(bool)
        
        return base_mask_fn
        
        
        
        
class SIRTask(UnstructuredTask):
    
    def __init__(self, time_start = 0,time_end = 50, eval_time_points=100, num_timepoints=20, backend: str = "jax") -> None:
        self.time_start = time_start
        self.time_end = time_end
        self.eval_time_points = eval_time_points
        self.num_timepoints = num_timepoints
        super().__init__("sir", sir,backend= backend, time_start = time_start,time_end = time_end, eval_time_points=eval_time_points, num_timepoints=num_timepoints)
        
        def ravel_meta_data(*meta_data):
            return jnp.concatenate((jnp.full((2,), jnp.nan), *meta_data))
        
        def unravel_meta_data(meta_data):
            return jnp.split(meta_data[2:], 4, axis=-1)
        
        def ravel_condition_mask(condition_mask):
            mask_theta, mask_theta3, mask_x0, mask_x1, mask_x2 = jnp.split(condition_mask, [2, 2 + self.num_timepoints,2 + 2*self.num_timepoints,2 + 3*self.num_timepoints], axis=-1)
            mask_theta3 = jnp.any(mask_theta3)[None]
            mask_x0 = jnp.any(mask_x0)[None]
            mask_x1 = jnp.any(mask_x1)[None]
            mask_x2 = jnp.any(mask_x2)[None]
            return jnp.concatenate((mask_theta, mask_theta3, mask_x0, mask_x1, mask_x2))
        
        def unravel_condition_mask(condition_mask):
            mask_theta, mask_theta3, mask_x0, mask_x1, mask_x2 = jnp.split(condition_mask, [2, 3,4,5], axis=-1)
            mask_theta3 = jnp.repeat(mask_theta3, self.num_timepoints, axis=-1)
            mask_x0 = jnp.repeat(mask_x0, self.num_timepoints, axis=-1)
            mask_x1 = jnp.repeat(mask_x1, self.num_timepoints, axis=-1)
            mask_x2 = jnp.repeat(mask_x2, self.num_timepoints, axis=-1)
            return jnp.concatenate((mask_theta,mask_theta3, mask_x0, mask_x1, mask_x2))
            
        self.ravel_meta_data = ravel_meta_data
        self.unravel_meta_data = unravel_meta_data
        self.ravel_condition_mask = ravel_condition_mask
        self.unravel_condition_mask = unravel_condition_mask
        
    def get_node_id(self):
        num_timepoints = self.dense_meta_data["x0"].shape[0]
        return jnp.array([0, 1] + [2] * num_timepoints + [3] * num_timepoints + [4] * num_timepoints + [5] * num_timepoints)
    
    def get_eval_node_id(self):
        return jnp.array([0, 1] + [2] * self.num_timepoints + [3] * self.num_timepoints + [4] * self.num_timepoints + [5] * self.num_timepoints)
    
    def get_x_dim(self):
        return 3* self.num_timepoints
    
    def get_theta_dim(self):
        return 2 + self.num_timepoints
    
    def sample_meta_data(self, key):
        key1, key2, key3, key4 = jrandom.split(key, 4)
        ts1 = jrandom.uniform(key1, (self.num_timepoints,), minval=self.time_start, maxval=self.time_end)
        ts2 = jrandom.uniform(key2, (self.num_timepoints,), minval=self.time_start, maxval=self.time_end)
        ts3 = jrandom.uniform(key3, (self.num_timepoints,), minval=self.time_start, maxval=self.time_end)
        ts4 = jrandom.uniform(key4, (self.num_timepoints,), minval=self.time_start, maxval=self.time_end)
        ts1 = jnp.sort(ts1)
        ts2 = jnp.sort(ts2)
        ts3 = jnp.sort(ts3)
        ts4 = jnp.sort(ts4)
        return (ts1, ts2, ts3, ts4)   
    
    
    def _get_conditional_sample_fn(self):
        
        @partial(jax.vmap, in_axes = [0, None, None, None])
        def sample_fn(key, condition_mask, x_o, meta_data, **kwargs):
            key_init, key_sample = jax.random.split(key, 2)
            init_vals_flat, potential_fn_wrapper = self._prepare_for_mcmc(key_init, condition_mask, x_o, meta_data)

            kernel = SliceKernel(step_size=0.3, num_steps=50)
            state = kernel.init_state(key,init_vals_flat)
            mcmc = MCMC(kernel, potential_fn_wrapper)
            samples, state = mcmc.run(state, 50000)
            return samples
        
        return sample_fn
    
    def _get_joint_sample_fn(self):
        @partial(jax.vmap, in_axes = [0, None])
        def sample_fn(key, meta_data, **kwargs):
            ts1, ts2, ts3, ts4 = self.unravel_meta_data(meta_data)
            samples = self.joint_sampler(key, ts1, ts2, ts3, ts4)
            return jnp.concatenate([samples[var] for var in self.var_names], axis=-1)

        return sample_fn
    
    
    def get_base_mask_fn(self):
        
        def base_mask_fn(node_id, meta_data):
            meta_data = meta_data.reshape(node_id.shape)
            max_size = node_id.shape[0]
            
            theta0_mask = node_id[:] == 0
            theta1_mask = node_id[:] == 1
            beta_mask = node_id[:] == 2
            I_mask = node_id[:] == 3
            R_mask = node_id[:] == 4
            D_mask = node_id[:] == 5


            in_past = meta_data[None,:] < meta_data[:, None]
            in_past_I_R = in_past & (I_mask[None, :] & R_mask[:, None])
            in_past_I_D = in_past & (I_mask[None, :] & D_mask[:, None])
            in_past_b_I = in_past & (beta_mask[None, :] & I_mask[:, None])
            # distances = jnp.abs(meta_data[None,:] - meta_data[:, None])
            # distances = distances / in_past
            # cross_distances_I_R = distances * (I_mask[None, :] & R_mask[:, None]) + ~(I_mask[None, :] & R_mask[:, None]) * jnp.inf
            # cross_distances_I_D = distances * (I_mask[None, :] & D_mask[:, None]) + ~(I_mask[None, :] & D_mask[:, None]) * jnp.inf
            # cross_distances_b_I = distances * (beta_mask[None, :] & I_mask[:, None]) + ~(beta_mask[None, :] & I_mask[:, None]) * jnp.inf
            # cross_distances_I_R = jnp.argmin(cross_distances_I_R, axis=1)
            # cross_distances_I_D = jnp.argmin(cross_distances_I_D, axis=1)
            # cross_distances_b_I = jnp.argmin(cross_distances_b_I, axis=1)

            index1 = jnp.where(I_mask, size=max_size)[0]
            index2 = jnp.where(R_mask, size=max_size)[0]
            index3 = jnp.where(D_mask, size=max_size)[0]


            mask_x = jnp.zeros((max_size, max_size), dtype=bool)
            # mask_x = mask_x.at[jnp.arange(max_size), cross_distances_I_R].set(True)
            # mask_x = mask_x.at[jnp.arange(max_size), cross_distances_I_D].set(True)
            # mask_x = mask_x.at[jnp.arange(max_size), cross_distances_b_I].set(True)
            mask_x = mask_x | in_past_I_R
            mask_x = mask_x | in_past_I_D
            mask_x = mask_x | in_past_b_I
            mask_x = mask_x.at[:, 0].set(False)
            mask_x = mask_x.at[0,:].set(False)
            mask_x = mask_x.at[0, 0].set(True)
            mask_x = mask_x | (jnp.tril(jnp.ones_like(mask_x)) & (beta_mask[None,:] & beta_mask[:,None]))
            mask_x = mask_x.at[index1, index1].set(True)
            mask_x = mask_x.at[index1[1:], index1[:-1]].set(True)
            mask_x = mask_x.at[index2, index2].set(True)
            mask_x = mask_x.at[index2[1:], index2[:-1]].set(True)
            mask_x = mask_x.at[index3, index3].set(True)
            mask_x = mask_x.at[index3[1:], index3[:-1]].set(True)
            mask_x = mask_x.at[0,:].set(False)
            mask_x  = mask_x | (jnp.eye(max_size, dtype=bool) * theta0_mask[None,:])
            mask_x  = mask_x | (jnp.eye(max_size, dtype=bool) * theta1_mask[None,:])
            mask_x = mask_x | (jnp.ones_like(mask_x) & (theta0_mask[None,:] & R_mask[:,None]))
            mask_x = mask_x | (jnp.ones_like(mask_x) & (theta1_mask[None,:] & D_mask[:,None]))
            mask_x = mask_x | (jnp.ones_like(mask_x) & (theta0_mask[None,:] & I_mask[:,None]))
            mask_x = mask_x | (jnp.ones_like(mask_x) & (theta1_mask[None,:] & I_mask[:,None]))

            return mask_x.astype(bool)
        
        return base_mask_fn
    