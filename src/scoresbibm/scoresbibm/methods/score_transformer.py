import jax
import jax.numpy as jnp

import optax

from functools import partial
from probjax.nn.loss_fn import denoising_score_matching_loss
from scoresbibm.methods.sde import init_sde_related
from scoresbibm.methods.models import AllConditionalScoreModel
from scoresbibm.methods.neural_nets import scalar_transformer_model
from scoresbibm.tasks.base_task import base_batch_sampler
from scoresbibm.utils.condition_masks import get_condition_mask_fn
from scoresbibm.utils.edge_masks import get_edge_mask_fn

def mean_std_per_node_id(data, node_ids):
    node_ids = node_ids.reshape(-1)
    mean = []
    std = []
    for i in range(node_ids.max()+1):
        index = jnp.where(node_ids == i)
        mean.append(jnp.mean(data[:,index]))
        std.append(jnp.std(data[:,index]))
    mean, std=  jnp.stack(mean), jnp.clip(jnp.stack(std), a_min=1e-2, a_max=None)
    mean = mean.reshape(-1,1)
    std = std.reshape(-1,1)
    return mean, std

def get_z_score_fn(data_mean_per_node_id, data_std_per_node_id):


    def z_score(data, node_id):
        shape = data.shape
        data = data.reshape(-1, len(node_id),  1)
        return ((data - data_mean_per_node_id[node_id])/data_std_per_node_id[node_id]).reshape(shape)

    def un_z_score(data, node_id):
        shape = data.shape
        data = data.reshape(-1, len(node_id),  1)
        return (data*data_std_per_node_id[node_id] + data_mean_per_node_id[node_id]).reshape(shape)
    return z_score, un_z_score


def run_train_transformer_model(
    key,
    params,
    opt_state,
    data,
    node_id,
    meta_data,
    total_number_steps,
    batch_size,
    update,
    batch_sampler,
    loss_fn,
    print_every=100,
    val_every=100,
    validation_fraction=0.05,
    val_repeat=2,
    val_error_ratio=1.1,
    stop_early_count=5,
):
    # Set up stuff for multi-device training
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices

    # Validation loss
    data_val, data_train = jnp.split(
        data, [max(int(validation_fraction * data.shape[0]), 0)], axis=0
    )
    data_val = jnp.repeat(
        data_val, val_repeat, axis=0
    )  # Multiple Monte Carlo samples for validation loss
    if meta_data is not None and meta_data.ndim > 2:
        meta_data_val, meta_data_train = jnp.split(
            meta_data, [max(int(validation_fraction * data.shape[0]), 1)], axis=0
        )
        meta_data_val = jnp.repeat(
            meta_data_val, val_repeat, axis=0
        )  # Multiple Monte Carlo samples for validation loss
    else:
        meta_data_val = meta_data
        meta_data_train = meta_data
    
    sampler = partial(
        batch_sampler, data=data_train, node_id=node_id, meta_data=meta_data_train, num_devices=num_devices
    )

    # Replicated for multiple devices
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * num_devices), params)
    replicated_opt_state = jax.tree_map(
        lambda x: jnp.array([x] * num_devices), opt_state
    )

    early_stopping_counter = 0

    l_val = None
    l_train = None
    min_l_val = 1e10
    early_stopping_params = None

    for j in range(total_number_steps):
        key, key_batch, key_update, key_val = jax.random.split(key, 4)
        data_batch, node_id_batch, meta_data_batch = sampler(key_batch, batch_size_per_device)
        loss, replicated_params, replicated_opt_state = update(
            replicated_params,
            replicated_opt_state,
            jax.random.split(key_update, (num_devices,)),
            data_batch,
            node_id_batch,
            meta_data_batch,
        )
        # Train loss
        if j == 0:
            l_train = loss[0]
        else:
            l_train = 0.9 * l_train + 0.1 * loss[0]

        # Validation loss
        if validation_fraction > 0 and ((j % val_every) == 0) and j > 50:
            l_val = loss_fn(
                jax.tree_map(lambda x: x[0], replicated_params),
                key_val,
                data_val,
                node_id,
                meta_data_val,
            )

            if l_val / l_train > val_error_ratio:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if l_val < min_l_val:
                min_l_val = l_val
                early_stopping_params = jax.tree_map(lambda x: x[0], replicated_params)

        if early_stopping_counter > stop_early_count:
            return early_stopping_params, jax.tree_map(
                lambda x: x[0], replicated_opt_state
            )

        # Print
        if (j % print_every) == 0:
            print("Train loss: ", l_train)
            if l_val is not None:
                print("Validation loss: ", l_val, early_stopping_counter)

    params = jax.tree_map(lambda x: x[0], replicated_params)
    opt_state = jax.tree_map(lambda x: x[0], replicated_opt_state)

    del replicated_opt_state
    del replicated_params

    return params, opt_state


def train_transformer_model(task, data, method_cfg, rng):
    device = method_cfg.device
    sde_params = dict(method_cfg.sde)
    model_params = dict(method_cfg.model)
    train_params = dict(method_cfg.train)

    # Data
    thetas, xs = data["theta"], data["x"]
    metadata = data.get("metadata", None)
    data = jnp.hstack([thetas, xs])
    node_id = task.get_node_id()
    theta_dim = task.get_theta_dim()
    x_dim = task.get_x_dim()
    data = data[..., None]
    if metadata is not None:
        metadata = metadata[..., None]
        
    # Z score 
    if method_cfg.train.z_score_data:
        mean_per_node_id, std_per_node_id = mean_std_per_node_id(data, node_id)
        z_score_fn, un_z_score_fn = get_z_score_fn(mean_per_node_id, std_per_node_id)
        data = z_score_fn(data, node_id)

    # Initialize stuff
    sde, T_min, T_max, _weight_fn, output_scale_fn = init_sde_related(
        data, name=sde_params.pop("name"), **sde_params
    )
    weight_fn = lambda t: _weight_fn(t).reshape(-1, 1, 1)
    if not model_params.pop("use_output_scale_fn", True):
        output_scale_fn = None
    init_fn, model_fn = scalar_transformer_model(
        theta_dim + x_dim, output_scale_fn=output_scale_fn, **model_params
    )

    rng, rng_init = jax.random.split(rng)
    params = init_fn(
        rng_init,
        jnp.ones((10,)),
        data[:10],
        node_id,
        jnp.zeros_like(data[:10]),
        meta_data=metadata,
    )

    # Training params
    total_number_steps = int(
        max(
            min(
                data.shape[0] * train_params["total_number_steps_scaling"],
                train_params["max_number_steps"],
            ),
            train_params["min_number_steps"],
        )
    )
    batch_size = train_params["training_batch_size"]

    print_every = total_number_steps // 10
    val_every = total_number_steps // train_params["val_every"]
    learning_rate = train_params["learning_rate"]
    schedule = optax.linear_schedule(
        learning_rate,
        train_params["min_learning_rate"],
        total_number_steps // 2,
        total_number_steps // 2,
    )
    optimizer = optax.chain(
        optax.adaptive_grad_clip(train_params["clip_max_norm"]), optax.adam(schedule)
    )
    opt_state = optimizer.init(params)

    condition_mask_params = dict(train_params["condition_mask_fn"])
    edge_mask_params = dict(train_params["edge_mask_fn"])

    # Get possible condition and edge mask functions
    condition_mask_fn = get_condition_mask_fn(
        condition_mask_params.pop("name", "structured"), **condition_mask_params
    )
    edge_mask_fn = get_edge_mask_fn(edge_mask_params["name"], task)

    # Training loop
    @jax.jit
    def loss_fn(params, key, data, node_id, meta_data):
        key_times, key_loss, key_condition = jax.random.split(key, 3)
        times = jax.random.uniform(
            key_times, (data.shape[0],), minval=T_min, maxval=T_max
        )

        # Structured conditioning
        condition_mask = condition_mask_fn(
            key_condition, data.shape[0], theta_dim, x_dim
        )
        if meta_data is None:
            edge_mask = jax.vmap(edge_mask_fn, in_axes=(None,0))(node_id, condition_mask)
        else:
            edge_mask = jax.vmap(edge_mask_fn, in_axes=(None, 0, 0))(node_id, condition_mask, meta_data)


        loss = denoising_score_matching_loss(
            params,
            key_loss,
            times,
            data,
            loss_mask=condition_mask,
            model_fn=model_fn,
            mean_fn=sde.marginal_mean,
            std_fn=sde.marginal_stddev,
            weight_fn=weight_fn,
            rebalance_loss = train_params["rebalance_loss"],
            data_id=node_id,
            condition_mask=condition_mask,
            meta_data=meta_data,
            edge_mask=edge_mask,
        )
        return loss

    @partial(jax.pmap, axis_name="num_devices")
    def update(params, opt_state, key, data, node_id, meta_data):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, data, node_id, meta_data)

        loss = jax.lax.pmean(loss, axis_name="num_devices")
        grads = jax.lax.pmean(grads, axis_name="num_devices")

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    rng, rng_train = jax.random.split(rng)
    batch_sampler = task.get_batch_sampler()
    params, opt_state = run_train_transformer_model(
        rng_train,
        params,
        opt_state,
        data,
        node_id,
        metadata,
        total_number_steps,
        batch_size,
        update,
        batch_sampler,
        loss_fn,
        print_every=print_every,
        val_every=val_every,
        validation_fraction=train_params["validation_fraction"],
        val_repeat=train_params["val_repeat"],
        stop_early_count=train_params["stop_early_count"],
    )

    sde_init_params = {
        "data": jax.device_put(data, jax.devices("cpu")[0]),
        **dict(method_cfg.sde),
    }
    model_init_params = {"num_nodes": theta_dim + x_dim, **dict(method_cfg.model)}
    edge_mask_params["task"] = task.name
    if method_cfg.train.z_score_data:
        z_score_params = {"mean_per_node_id": mean_per_node_id, "std_per_node_id": std_per_node_id, "z_score_fn": z_score_fn, "un_z_score_fn": un_z_score_fn}
    else:
        z_score_params = None
    model = AllConditionalScoreModel(
        params,
        model_fn,
        sde,
        sde_init_params=sde_init_params,
        model_init_params=model_init_params,
        edge_mask_fn_params=edge_mask_params,
        z_score_params=z_score_params,
    )
    # Posterior as default
    default_conditon_mask = jnp.array([0] * theta_dim + [1] * x_dim, dtype=jnp.bool_)
    model.set_default_condition_mask(default_conditon_mask)
    model.set_default_node_id(node_id)
    model.set_default_edge_mask_fn(edge_mask_fn)

    return model
