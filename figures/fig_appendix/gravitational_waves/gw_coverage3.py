# %%
import jax 
import jax.numpy as jnp

import haiku as hk

import torch

import matplotlib.pyplot as plt

from scoresbibm.tasks import get_task
from scoresbibm.methods.score_transformer import run_train_transformer_model
from scoresbibm.methods.neural_nets import scalar_transformer_model
from scoresbibm.methods.sde import init_sde_related
from scoresbibm.utils.plot import use_style


def main():
    print("Start")
    # %%
    thetas = torch.load("/mnt/qb/macke/mgloeckler90/thetas_0.pt")
    theta1 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_1.pt")
    theta2 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_2.pt")
    theta3 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_3.pt")
    theta4 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_4.pt")
    theta5 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_5.pt")
    theta6 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_6.pt")
    theta7 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_7.pt")
    theta8 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_8.pt")
    theta9 = torch.load("/mnt/qb/macke/mgloeckler90/thetas_9.pt")


    # %%
    xs_raw = torch.load("/mnt/qb/macke/mgloeckler90/xs_0.pt")
    xs_raw1 = torch.load("/mnt/qb/macke/mgloeckler90/xs_1.pt")
    xs_raw2 = torch.load("/mnt/qb/macke/mgloeckler90/xs_2.pt")
    xs_raw3 = torch.load("/mnt/qb/macke/mgloeckler90/xs_3.pt")
    xs_raw4 = torch.load("/mnt/qb/macke/mgloeckler90/xs_4.pt")
    xs_raw5 = torch.load("/mnt/qb/macke/mgloeckler90/xs_5.pt")
    xs_raw6 = torch.load("/mnt/qb/macke/mgloeckler90/xs_6.pt")
    xs_raw7 = torch.load("/mnt/qb/macke/mgloeckler90/xs_7.pt")
    xs_raw8 = torch.load("/mnt/qb/macke/mgloeckler90/xs_8.pt")
    xs_raw9 = torch.load("/mnt/qb/macke/mgloeckler90/xs_9.pt")

    # %%
    thetas = torch.cat([thetas, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9], dim=0)
    xs_raw = torch.cat([xs_raw, xs_raw1, xs_raw2, xs_raw3, xs_raw4, xs_raw5, xs_raw6, xs_raw7, xs_raw8, xs_raw9], dim=0)

    # %%
    thetas = jnp.array(thetas.numpy())
    xs_raw = jnp.array(xs_raw.numpy())

    thetas = jax.device_put(thetas, jax.devices("cpu")[0])
    xs_raw = jax.device_put(xs_raw, jax.devices("cpu")[0])

    # %%
    thetas_mean = jnp.mean(thetas, axis=0, keepdims=True)
    thetas_std = jnp.std(thetas, axis=0, keepdims=True)

    thetas = (thetas - thetas_mean) / thetas_std

    # %%
    def z_score_thetas(thetas):
        thetas = (thetas - thetas_mean) / thetas_std
        return thetas

    def un_z_score_thetas(thetas):
        thetas = thetas * thetas_std + thetas_mean
        return thetas

    # %%
    xs_mean = jnp.mean(xs_raw, axis=0, keepdims=True)
    xs_mean = jnp.mean(xs_mean, axis=-1, keepdims=True)
    xs_std = jnp.std(xs_raw, axis=0, keepdims=True)
    xs_std = jnp.std(xs_std, axis=-1, keepdims=True)

    xs = (xs_raw - xs_mean) / xs_std

    # %%
    def z_score_xs(xs):
        xs = (xs - xs_mean) / xs_std
        return xs

    def un_z_score_xs(xs):
        xs = xs * xs_std + xs_mean
        return xs

    # %%
    import numpy as np
    def ravel(data, axis=-1):
        flat_data, tree = jax.tree_flatten(data)
        split_dims = np.cumsum(np.array([flat.shape[axis] for flat in flat_data]))[:-1]
        flat_data = jnp.concatenate(flat_data, axis=axis)
        def unravel(flat_data):
            flat_data = jnp.split(flat_data, split_dims, axis=axis)
            flat_data = jax.tree_unflatten(tree, flat_data)
            return flat_data
        def unflatten(flat_data):
            flat_data = jnp.split(flat_data, split_dims, axis=axis)
            return flat_data 
        return flat_data, unravel, unflatten

    # %%
    data = {"theta0": thetas[:, 0][:, None], "theta1": thetas[:, 1][:, None], "x0": xs[:, 0], "x1": xs[:, 1]}
    data_flat, unravel, unflatten = ravel(data)
    theta_dim = 2
    x_dim = data_flat.shape[-1] - theta_dim
    node_id = jnp.arange(0, 4)
    condition_mask = jnp.array([False]*theta_dim + [True]*x_dim)

    # %%

    sde, T_min, T_max, weight_fn, output_scale_fn =  init_sde_related(data_flat, "vesde", sigma_min=1e-5, sigma_max=15.)

    # %%
    from probjax.nn.helpers import GaussianFourierEmbedding
    from probjax.nn.transformers import Transformer


    token_dim = 100
    id_dim = 40
    cond_dim = 10

    embedding_net1 = lambda x: jnp.repeat(x, token_dim, axis=-1)

    embedding_net2 = lambda x:hk.Sequential([
            hk.Conv1D(output_channels=16, kernel_shape=9, stride=2, padding='SAME', name='conv1'), # 4000
            jax.nn.gelu,
            hk.Conv1D(output_channels=32, kernel_shape=6, stride=2, padding='SAME', name='conv2'), # 2000
            jax.nn.gelu,
            hk.Conv1D(output_channels=64, kernel_shape=3, stride=2, padding='SAME', name='conv3'), # 1000
            jax.nn.gelu,
            hk.Conv1D(output_channels=128, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 500
            jax.nn.gelu,
            hk.Conv1D(output_channels=256, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 250
            jax.nn.gelu,
            hk.Conv1D(output_channels=512, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 125
            jax.nn.gelu,
            hk.Conv1D(output_channels=512, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 64
            jax.nn.gelu,
            hk.Conv1D(output_channels=512, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 32
            jax.nn.gelu,
            hk.Conv1D(output_channels=512, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 16
            jax.nn.gelu,
            hk.Conv1D(output_channels=512, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 8
            jax.nn.gelu,
            hk.Conv1D(output_channels=512, kernel_shape=3, stride=2, padding='SAME', name='conv4'), # 4
            jax.nn.gelu,
            hk.Flatten(),
            hk.Linear(token_dim, name='linear'),
        ])(x)[..., None,:]

    embedding_nets = [embedding_net1,embedding_net1, embedding_net2, embedding_net2]
    output_fn = [lambda x: hk.Linear(1)(x), lambda x: hk.Linear(1)(x), lambda x: hk.Linear(8192)(x), lambda x: hk.Linear(8192)(x)]


    def model(t, data, data_id, condition_mask, edge_mask=None):
        
        data = unflatten(data)

        assert len(data) == len(embedding_nets), "Number of data elements and embedding nets must match"
        
        data_embedded = jax.tree_map(lambda x, net: net(x[..., :,None]), data, embedding_nets)

        data_embedded = jnp.concatenate(data_embedded, axis=-2)
        
        _, current_nodes, _ = data_embedded.shape
        
        time_embeder = GaussianFourierEmbedding(128)
        id_embedder= hk.Embed(4, id_dim)
        
        id_embedding = id_embedder(data_id)
        id_embedding = jnp.broadcast_to(
            id_embedding, data_embedded.shape[:-1] + (id_dim,)
        )
        tokens = jnp.concatenate([data_embedded, id_embedding], axis=-1)
        time = time_embeder(t[..., None])
        
        condition_token = hk.get_parameter(
            "condition_token",
            shape=[1, 1, cond_dim],
            init=hk.initializers.RandomNormal(
                0.01, 0.
            ),
        )
        condition_mask = unflatten(condition_mask)
        condition_mask = jax.tree_map(lambda x: jnp.any(x, axis=-1, keepdims=True), condition_mask)
        condition_mask = jnp.concatenate(condition_mask, axis=-1)
        condition_mask = condition_mask.reshape(-1, current_nodes, 1)
        condition_token = condition_mask * condition_token
    
        condition_token = jnp.broadcast_to(
            condition_token, tokens.shape[:-1] + (cond_dim,)
        )
        tokens = jnp.concatenate([tokens, condition_token], -1)

        
        model = Transformer(
            num_heads=4,
            num_layers=8,
            attn_size=20,
            widening_factor=3,
        )
        
        h = model(tokens, context=time, mask=edge_mask)
        out = jnp.split(h, current_nodes, axis=-2)
        out = jax.tree_map(lambda x, fn: fn(x), out, output_fn)
        out = jnp.concatenate(out, axis=-1)
        out = jnp.squeeze(out, axis=-2)
        out = output_scale_fn(t, out)

        return out
        
        
    init_fn, model_fn = hk.without_apply_rng(hk.transform(model))


    # %%
    params = init_fn(jax.random.PRNGKey(42), jnp.ones((1,)), data_flat[:10], node_id, condition_mask)

    # %%
    out = model_fn(params, jnp.ones((1,))*0.01, data_flat[:10], node_id, condition_mask)



    # %%
    params = jnp.load("only_post_params.pt.npy", allow_pickle=True).item()
    # unpack the parameters
    print("Loaded model")

    # %%
    end_mean = sde.marginal_mean(jnp.array(1.0))[None, ...]
    end_mean = jax.device_put(end_mean, jax.devices()[0])

    # %%
    end_std = sde.marginal_stddev(jnp.array(1.0))[None, ...]
    end_std = jax.device_put(end_std, jax.devices()[0])

    # %%
    from functools import partial
    from probjax.utils.sdeint import sdeint


    # Reverse SDE drift
    def drift_backward(t, x, node_ids=None, condition_mask=None, edge_mask=None, score_fn = model_fn):
        score = score_fn(params, t.reshape(-1, 1), x[None,...], node_ids,condition_mask, edge_mask=edge_mask)
        score = score.reshape(x.shape)

        f =  sde.drift(t,x) - sde.diffusion(t,x)**2 * score
        f = f * (1-condition_mask)
        
        return f.reshape(x.shape)

    # Reverse SDE diffusion
    def diffusion_backward(t,x, **kwargs):
        #t = T - t
        b =  sde.diffusion(t,x) 
        b = b * (1-condition_mask)
        return b.reshape(x.shape)

    # %%



    def sample_fn(key, shape, node_ids,condition_mask, condition_value, time_steps=1000, edge_mask=None, score_fn=model_fn):

        key1, key2 = jax.random.split(key, 2)
        # Sample from noise distribution at time 1
        x_T = jax.random.normal(key1, shape + (condition_value.shape[0],)) * end_std + end_mean

        
        x_T = x_T * (1-condition_mask) + condition_value * condition_mask
        # Solve backward sde
        keys = jax.random.split(key2, shape)
        ys = jax.vmap(lambda *args: sdeint(*args, noise_type="diagonal", only_final=True), in_axes= (0, None, None, 0, None), out_axes=0)(keys, lambda t, x: drift_backward(t, x, node_ids, condition_mask, edge_mask=edge_mask, score_fn=score_fn), lambda t, x: diffusion_backward(t, x), x_T, jnp.linspace(1.,T_min, time_steps))
        return ys

    # %%
    thetas_test = torch.load("/mnt/qb/macke/mgloeckler90/thetas_10.pt")
    thetas_test = jnp.array(thetas_test.numpy())
    thetas_test = z_score_thetas(thetas_test)
    thetas_test = jax.device_put(thetas_test, jax.devices()[0])

    # %%
    xs_test = torch.load("/mnt/qb/macke/mgloeckler90/xs_10.pt")
    xs_test = jnp.array(xs_test.numpy())
    xs_test = z_score_xs(xs_test).reshape(xs_test.shape[0],-1)
    xs_test = jax.device_put(xs_test, jax.devices()[0])

    # %%
    data_test_flat = jnp.concatenate([thetas_test, xs_test], axis=-1)

    # %%
    thetas_mean = jax.device_put(thetas_mean, jax.devices()[0])
    thetas_std = jax.device_put(thetas_std, jax.devices()[0])

    # %%
    edge_mask = jnp.ones((4,4), dtype=bool)
    edge_mask1 = edge_mask.at[:, 2].set(False)
    edge_mask1 = edge_mask1.at[2, :].set(False)

    edge_mask2 = edge_mask.at[:, 3].set(False)
    edge_mask2 = edge_mask2.at[3, :].set(False)

    # %%
    ((thetas_test[:, 0] - thetas_test[:, 1])**2).argsort()

    # %%
    index = 9001
    x_o = jax.device_put(data_test_flat[index], jax.devices()[0])

    # %%
    # Reverse SDE drift
    def drift_ode(t, x, x_o,node_ids=node_id, condition_mask=np.array(condition_mask), edge_mask=edge_mask2, score_fn = model_fn):
        x = x_o.at[~condition_mask].set(x)
        score = score_fn(params, t.reshape(-1, 1), x[None,...], node_ids,condition_mask, edge_mask=edge_mask)
        score = score.reshape(x.shape)

        f =  sde.drift(t,x) - 0.5*sde.diffusion(t,x)**2 * score
        f = f * (1-condition_mask)
        
        return f[~condition_mask]


    # %%
    from probjax.utils.odeint import odeint
    from probjax.distributions import Normal, Independent
    from probjax.distributions.transformed_distribution import TransformedDistribution

    # %%
    x_o = jax.device_put(data_test_flat[index], jax.devices()[0])
    m0 = jax.device_put(end_mean[0,~condition_mask], jax.devices()[0])
    s0 = jax.device_put(end_std[0,~condition_mask], jax.devices()[0])
    q0 = Independent(Normal(m0,s0),1)

    # %%
    @jax.jit
    def sample_post(key, x_o):
        thetas =  sample_fn(key, (500,), node_id, condition_mask, x_o, edge_mask=edge_mask2)[:, :2]
        return un_z_score_thetas(thetas)

    # %%
    @jax.jit
    def log_probs_post(thetas, x_o):
        def f(x_T):
            y = odeint(lambda t,x: drift_ode(t,x,x_o, edge_mask=edge_mask2), x_T, jnp.linspace(1e-5,1, 100)[::-1], method="rk3")[-1]
            y = un_z_score_thetas(y)
            return y

        q = TransformedDistribution(q0, f)
        return q.log_prob(thetas)

    # %%
    log_probs_q = []
    log_probs_true = []

    for i in range(500):
        print("Computing", i)
        x_o = jax.device_put(data_test_flat[i], jax.devices()[0])
        theta_true = un_z_score_thetas(x_o[:2])
        key = jax.random.PRNGKey(i)
        thetas = sample_post(key, x_o)
        log_prob = log_probs_post(thetas, x_o)
        log_prob_true = log_probs_post(theta_true, x_o)
        
        log_probs_q.append(log_prob)
        log_probs_true.append(log_prob_true)
        

    # %%
    log_probs_q_vec = jnp.stack(log_probs_q)
    log_probs_true_vec = jnp.stack(log_probs_true)
    
    jnp.save("log_probs_q_vec_partial2_posterior.npy", log_probs_q_vec)
    jnp.save("log_probs_true_vec_partial2_posterior.npy", log_probs_true_vec)

 
 
if __name__ == "__main__":
    main()