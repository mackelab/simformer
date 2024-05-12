
import jax
import jax.numpy as jnp

from probjax.utils.odeint import _odeint
from probjax.utils.sdeint import sdeint
from scoresbibm.tasks.all_conditional_tasks import AllConditionalBMTask




ts = jnp.linspace(0, 3, 100)


V0 = -65.0  # Initial membrane voltage (mV)


def I_inj_fn(t, t_val, Vt_val):
    index = jnp.searchsorted(t_val, t, side='right') - 1
    return jax.lax.cond(index < 0, lambda _: 0.0, lambda _: Vt_val[index], None)

def efun(x):
    return jnp.where(x < 1e-4, 1-x /2, x/(jnp.exp(x) - 1.0))

def alpha_m_fn(V):
    return 0.32 * efun(-0.25*(V - V0 - 13.0)) / 0.25

def beta_m_fn(V):
    return 0.28 * efun(0.2*(V - V0 - 40.0)) / 0.2

def alpha_h_fn(V):
    return 0.128 * jnp.exp(-(V - V0 - 17.0) / 18.0)

def beta_h_fn(V):
    return 4.0 / (1.0 + jnp.exp(-(V - V0 - 40.0) / 5.0))

def alpha_n_fn(V):
    return 0.032 * efun(-0.2*(V - V0 - 15.0)) / 0.2

def beta_n_fn(V):
    return 0.5 * jnp.exp(-(V - V0 - 10.0) / 40.0)

def hodgkin_huxley(t, state, Cm, g_Na, g_K, g_L, E_Na, E_K, E_L, t_val,Vt_val):
    V, m, h, n, H = state

    # Membrane capacitance (uF/cm^2)
    # Sodium conductance (mS/cm^2)
    # Potassium conductance (mS/cm^2)
    # Leak conductance (mS/cm^2)
    # Sodium reversal potential (mV)
    # Potassium reversal potential (mV)
    # Leak reversal potential (mV)

    # Alpha and beta functions for m, h, and n
    alpha_n = alpha_n_fn(V)
    beta_n = beta_n_fn(V)
    alpha_m = alpha_m_fn(V)
    beta_m = beta_m_fn(V)
    alpha_h = alpha_h_fn(V)
    beta_h = beta_h_fn(V)
    
    i_Na = g_Na * m**3 * h * (V - E_Na)
    i_K = g_K * n**4 * (V - E_K)
    i_L = g_L * (V - E_L)
    # Hodgkin-Huxley ODEs
    dV = (I_inj_fn(t, t_val, Vt_val) - i_Na - i_K - i_L) / Cm
    dm = alpha_m * (1.0 - m) - beta_m * m
    dh = alpha_h * (1.0 - h) - beta_h * h
    dn = alpha_n * (1.0 - n) - beta_n * n
    #dH = (V*I_inj_fn(t, t_val, Vt_val) - i_Na * (V - E_Na) - i_K * (V - E_K) - i_L * (V - E_L)) / 10**6 # in mJ
    dH = i_Na  # in mJ


    return dV, dm, dh, dn, dH

diffusion_fn = lambda t, state, *args: jnp.array([0.05, 0., 0., 0., 0.]) # Voltage noise



def compute_summary_statistics(V, ts_dense, t_val):
    V_mod = jnp.where(V < -10, -10, V)
    V_mod = jnp.where(jnp.diff(V, prepend=jnp.array([-10.])) < 5., 0, 1)
    V_mod = jnp.convolve(V_mod, jnp.ones(20), mode='same')
    V_mod = jnp.where(V_mod > 1, 1, 0)
    spike_occured = jnp.where(jnp.diff(V_mod, prepend=jnp.array([-10.])) < 0, 1., 0.)
    spike_count = jnp.sum(spike_occured, keepdims=True)

    # Resting potential
    V_rest = jnp.where(ts_dense < t_val[0], V, jnp.nan)
    V_rest_mean = jnp.nanmean(V_rest, keepdims=True)
    V_rest_std = jnp.nanstd(V_rest, keepdims=True)

    # Spiking domain mean 
    V_spike = jnp.where((ts_dense > t_val[0]) & (ts_dense < t_val[-1]), V, jnp.nan)    
    V_spike_mean = jnp.nanmean(V_spike)
    # Moments 
    n_mom = 4
    input_voltage_mask = (ts_dense > t_val[0]) & (ts_dense < t_val[-1])
    V_input = jnp.where(input_voltage_mask, V, jnp.nan)
    std_input_voltage = jnp.nanstd(V_input)
    std_pw = jnp.power(std_input_voltage, jnp.linspace(3, n_mom, n_mom - 2))
    std_pw = jnp.concatenate((jnp.ones(1), std_pw))

    moments = jnp.linspace(2, n_mom, n_mom - 1)
    V_input_mean = jnp.nanmean(V_input)
    V_input_pw = V_input - V_input_mean
    V_input_pw = jnp.power(V_input_pw[...,None], moments)
    V_input_moments = jnp.nanmean(V_input_pw, axis=0)
    V_input_moments = V_input_moments / std_pw

    return spike_count, V_rest_mean, V_rest_std, V_spike_mean[...,None], *jnp.split(V_input_moments, n_mom - 1, axis=-1)


def hh_model(t_min=0., t_max=200., t_val=jnp.array([50., 150.]), Vt_val=jnp.array([4., 0.]), V0 = -65.0, return_voltage=False, voltage_based_energy=False):
    
    ts_dense = jnp.linspace(t_min, t_max, 5000)
    m0 = alpha_n_fn(V0) / (alpha_n_fn(V0) + beta_n_fn(V0))
    h0 = alpha_h_fn(V0) / (alpha_h_fn(V0) + beta_h_fn(V0))
    n0 = alpha_m_fn(V0) / (alpha_m_fn(V0) + beta_m_fn(V0))
    H0 = jnp.zeros_like(V0)

    var_names = ["theta0", "theta1", "theta2", "theta3", "theta4", "theta5", "theta6", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    def joint_sampler(key):
        key1, key2, key3, key4, key5, key6, key7, key_sim = jax.random.split(key, 8) 
        Cm = jax.random.uniform(key1, (), minval=1., maxval=2.)
        g_Na = jax.random.uniform(key2, (), minval=60, maxval=120)
        g_K = jax.random.uniform(key3, (), minval=10, maxval=30)
        g_L = jax.random.uniform(key4, (), minval=0.1, maxval=0.5)
        E_Na = jax.random.uniform(key5, (), minval=40, maxval=70)
        E_K = jax.random.uniform(key6, (), minval=-100, maxval=-60)
        E_V = jax.random.uniform(key7, (), minval=-90, maxval=-60)
        params = (Cm, g_Na, g_K, g_L, E_Na, E_K, E_V)
        vals = sdeint(key_sim,hodgkin_huxley, diffusion_fn, (V0, m0, h0, n0, H0), ts_dense, *params, t_val, Vt_val, noise_type="diagonal")
        V = vals[...,0]
        H = vals[...,-1]
        V = jnp.nan_to_num(V, nan=V0, posinf=V0, neginf=V0)
        V = jnp.clip(V, -100, 100)
        summary_stats = compute_summary_statistics(V, ts_dense, t_val)
        total_energy = H[-1][...,None]
        parmas =  {"theta0": Cm[None], "theta1": g_Na[None], "theta2": g_K[None], "theta3": g_L[None], "theta4": E_Na[None], "theta5": E_K[None], "theta6": E_V[None]}
        data = {"x{i}".format(i=i):val for i, val in enumerate(summary_stats)}
        energys =  { f"x{len(data)}": total_energy}
        return {**parmas, **data,**energys}
        

    return var_names, joint_sampler, None


class HHTask(AllConditionalBMTask):
    
    def __init__(self, backend: str = "jax", t_min=0., t_max=200., t_val=jnp.array([50., 150.]), Vt_val=jnp.array([4., 0.]), V0 = -65.0, return_voltage=False, voltage_based_energy=False) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.t_val = t_val
        self.Vt_val = Vt_val
        self.V0 = V0
        self.return_voltage = return_voltage
        self.voltage_based_energy = voltage_based_energy
        super().__init__("hh", hh_model, backend)
        
    
    def get_simulator(self):
        def simulator(key, theta):
            ts = jnp.linspace(self.t_min, self.t_max, 5000)
            Cm, g_Na, g_K, g_L, E_Na, E_K, E_V = theta
            m0 = alpha_n_fn(self.V0) / (alpha_n_fn(self.V0) + beta_n_fn(self.V0))
            h0 = alpha_h_fn(self.V0) / (alpha_h_fn(self.V0) + beta_h_fn(self.V0))
            n0 = alpha_m_fn(self.V0) / (alpha_m_fn(self.V0) + beta_m_fn(self.V0))
            H0 = jnp.zeros_like(self.V0)
            params = (Cm, g_Na, g_K, g_L, E_Na, E_K, E_V)
            vals = sdeint(key,hodgkin_huxley, diffusion_fn, (self.V0, m0, h0, n0, H0), ts, *params, self.t_val, self.Vt_val, noise_type="diagonal")
            V = vals[...,0]
            H = vals[...,-1]
            V = jnp.nan_to_num(V, nan=self.V0, posinf=self.V0, neginf=self.V0)
            V = jnp.clip(V, -100, 100)
            summary_stats = compute_summary_statistics(V, ts, self.t_val)
            return V, H, jnp.concatenate(summary_stats, axis=-1)
        return simulator
                
        
    def get_data(self, num_samples: int, rng=None, batch_size=5000):
        rounds = num_samples // batch_size + 1
        batch_size = min(batch_size, num_samples)
        thetas = []
        xs = []
        for _ in range(rounds):
            rng, key = jax.random.split(rng)
            keys = jax.random.split(key, (batch_size,))
            samples = jax.vmap(self.joint_sampler)(keys)
            theta = jnp.concatenate(
                [samples[var] for var in self.var_names if "theta" in var], axis=-1
            )
            x = jnp.concatenate(
                [samples[var] for var in self.var_names if "x" in var], axis=-1
            )
            thetas.append(theta)
            xs.append(x)
        thetas = jnp.concatenate(thetas, axis=0)[:num_samples, ...]
        xs = jnp.concatenate(xs, axis=0)[:num_samples, ...]

        return {"theta":thetas, "x":xs}
    
    def get_output_transform(self):
        def output_transform_fn(x, node_id, meta_data=None):
            rounded_x = jnp.round(x)
            x = jnp.where(node_id == 7, rounded_x, x)
            return x
        return output_transform_fn
    
    def get_base_mask_fn(self):
        pass 
    
    def _get_conditional_sample_fn(self):
        def sample_fn(*args, **kwargs):
            raise NotImplementedError()
        return sample_fn
    
