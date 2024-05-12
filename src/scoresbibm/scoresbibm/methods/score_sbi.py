

import jax 
import jax.numpy as jnp


import optax

from probjax.nn.tokenizer import scalarize
from probjax.nn.loss_fn import denoising_score_matching_loss

from probjax.distributions import Normal
from probjax.distributions.transformed_distribution import TransformedDistribution


from functools import partial
from scoresbibm.methods.neural_nets import conditional_mlp
from scoresbibm.methods.models import ScorePosteriorModel

from scoresbibm.methods.sde import init_sde_related



def run_train_conditional_score_model(key, params, opt_state, data, num_epochs, num_steps, batch_size,  update, print_every=100):
    """ Runs the training loop for the conditional score model. Assumes update is compiled using jax.pmap."""
    
    # Set up stuff for multi-device training
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices
    # Replicated for multiple devices
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * num_devices), params)
    replicated_opt_state = jax.tree_map(lambda x: jnp.array([x] * num_devices), opt_state)

    for j in range(num_epochs):
        l = 0
        for i in range(num_steps):
            key, key_batch, key_update = jax.random.split(key, 3)
            data_batch = jax.random.choice(key_batch,data, shape=(num_devices, batch_size_per_device, ), axis=0, replace=True)
            loss, replicated_params, replicated_opt_state = update(replicated_params, replicated_opt_state, jax.random.split(key_update, (num_devices,)), data_batch)
            l += loss[0] /num_steps
        if (j % print_every) == 0:     
            print("Train loss: ",l)
            
    params = jax.tree_map(lambda x: x[0], replicated_params)
    
    return params


# Set up stuff for multi-device training
def train_conditional_score_model(task, thetas,xs, method_cfg, rng):

    device = method_cfg.device
    sde_params = dict(method_cfg.sde)
    model_params = dict(method_cfg.model)
    train_params = dict(method_cfg.train)
    
    
    # Data
    data = jnp.hstack([thetas, xs])
    theta_dim = thetas.shape[-1]
    x_dim = xs.shape[-1]
    
    # Initialize stuff
    sde, T_min,T_max, weight_fn, output_scale_fn = init_sde_related(thetas, name = sde_params.pop("name"), **sde_params)
    if not model_params.pop("use_output_scale_fn", True):
        output_scale_fn = None
    init_fn, model_fn = conditional_mlp(theta_dim, output_scale_fn=output_scale_fn, **model_params)
    
    rng, rng_init = jax.random.split(rng)
    params = init_fn(rng_init, jnp.ones((10,)), thetas[:10], xs[:10])


    # Training params
    total_number_steps = int(data.shape[0] * train_params["total_number_steps_scaling"])
    batch_size = train_params["training_batch_size"]
    num_steps = data.shape[0] // batch_size + 1

    num_epochs = min(total_number_steps // num_steps + 1, train_params["max_num_epochs"])
    total_number_steps = num_epochs * num_steps
    print_every = num_epochs // 10
    learning_rate = train_params["learning_rate"]
    schedule = optax.linear_schedule(learning_rate, 0., total_number_steps//2, total_number_steps//2)
    optimizer = optax.chain(optax.adaptive_grad_clip(train_params["clip_max_norm"]), optax.adam(schedule))
    opt_state = optimizer.init(params)
    
    
    # Training loop
    @jax.jit
    def loss_fn(params, key, data):
        thetas, xs = jnp.split(data, [theta_dim,], axis=-1)
        key_times, key_loss = jax.random.split(key,2)
        times = jax.random.uniform(key_times, (data.shape[0],), minval=T_min, maxval =T_max)
        loss = denoising_score_matching_loss(params, key_loss, times, thetas, None, xs, model_fn = model_fn, mean_fn = sde.marginal_mean, std_fn=sde.marginal_stddev, weight_fn=weight_fn, axis=-1)
        return loss

    @partial(jax.pmap, axis_name="num_devices")
    def update(params, opt_state, key, data):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, data)

        loss = jax.lax.pmean(loss, axis_name="num_devices")  # Syncs loss
        grads = jax.lax.pmean(grads, axis_name="num_devices") # Syncs grads
        
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    
    rng, rng_train = jax.random.split(rng)
    params = run_train_conditional_score_model(rng_train, params, opt_state, data, num_epochs, num_steps, batch_size, update, print_every=print_every)
    
    sde_init_params = {"data": thetas, **sde_params}
    model_init_params = {"output_dim": theta_dim, **model_params}
    model = ScorePosteriorModel(params, model_fn, sde, sde_init_params=sde_init_params, model_init_params=model_init_params)
   
    return model





    



