import functools
import itertools

import jax
from jax import numpy as jnp

import config
from src.utils.misc_utils import is_prob_mass

# For V^*
def v_iteration(env, n_iterations=1000):
    v = jnp.zeros(env.state_space)
    for _ in range(n_iterations):
        v_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, v)).max(1)
        v = v_new.clone()
    return v

# For Q^*
def q_iteration(env, n_iterations=1000, eps=1e-6):
    q = jnp.zeros((env.state_space, env.action_space))
    for _ in range(n_iterations):
        q_star = q.max(1)
        q_new = (env.R + jnp.einsum("xay,y->xa", env.gamma * env.P, q_star))
        q = q_new.clone()
    return q

# For V^pi
@jax.partial(jax.jit, static_argnums=(0,))
def get_value(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    r_pi = jnp.einsum('xa,xa->x', env.R, pi)
    v = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi), r_pi)
    return v

# For d^pi
@jax.partial(jax.jit, static_argnums=(0,))
def get_dpi(env, pi):
    p_pi = jnp.einsum('xay,xa->xy', env.P, pi)
    rho = env.p0
    d_pi = jnp.linalg.solve((jnp.eye(env.state_space) - env.gamma * p_pi.T), (1 - env.gamma) * rho)
    d_pi /= d_pi.sum()
    return d_pi

# For Q^pi
@jax.partial(jax.jit, static_argnums=(0,))
def get_q_value(env, pi, v):
    v = get_value(env, pi)
    q_pi = jnp.einsum('xay,y->xa', env.P, v)
    q = env.R + env.gamma * q_pi
    return q

# For A^pi
@jax.partial(jax.jit, static_argnums=(0,))
def get_adv(env, pi):
    v_k = get_value(env, pi)
    q = get_q_value(env, pi, v_k)
    adv = q - jnp.expand_dims(v_k, 1)
    return adv

# For KL(p || q)
def kl_fn(weight, p, q):
    _kl = ((p + 1e-6) * jnp.log((p + 1e-6) / (q + 1e-6))).sum(1)
    _kl = (weight * _kl).sum()
    return jnp.clip(_kl, 0.)


@jax.jit
def softmax_ppo(pi_old, adv, eta, clip=0.):
    pi = pi_old * (1 + eta * adv) # Exact FMAPG update
    pi = jnp.clip(pi, a_min=clip) # clip negative values
    pi = pi / pi.sum(1, keepdims=True) # normalize
    return pi


@jax.jit
def mdpo(pi, adv, eta): # MDPO = FMAPG with direct, but A instead of Q
    pi = pi * (jnp.exp(eta * adv)) # similar to exponential weights
    denom = pi.sum(1, keepdims=True)
    pi = pi / denom # normalize
    return pi

######### Don't understand this one. Why is it called PG, but returns loss? # Needs to be renamed. Computes the FMAPG loss. 
def pg(pi, pi_old, d_s, adv): 
    loss = (pi_old * jnp.log(pi) * adv).sum(1)
    loss = (d_s * loss).sum()
    return loss
#####

def ppo(pi, pi_old, d_s, adv):  # computes the MDPO loss
    loss = jnp.einsum('s, sa->a', d_s, pi * adv).sum()
    return loss

# Above functions assume that the A/Q functions are computed exactly. 
# In each iteration. No off-policy sampling / importance sampling ratios. 

# For H(\pi)
@jax.partial(jax.jit, static_argnums=(0,))
def entropy_fn(env, pi):
    d_s = get_dpi(env, pi)
    entropy = - (pi * jnp.log(pi + 1e-6)).sum(1)
    out = jnp.einsum('s,sa', d_s, entropy)
    return out

# For exact policy improvement
def pi_improve(pi_fn, env, pi_old, eta):
    pi = pi_old
    adv = get_adv(env, pi_old) # compute advantage
    pi = pi_fn(pi, adv, eta) # update

    # Only for logging, and maybe use the KL for termination
    v_kp1 = get_value(env, pi) # compute value
    kl = kl_fn(get_dpi(env, pi), pi, pi_old) 

    stats = {
        "pi/return": (env.p0 @ v_kp1),
        "pi/kl": kl
    }
    return pi, v_kp1, stats


def approx_pi_improve(env, pi_fn, iterations=10):
    eval_pi = functools.partial(get_value, env)

    def loss_fn(pi, pi_old, d_s, adv):
        loss = pi_fn(pi, pi_old, d_s, adv)
        kl = config.eta * kl_fn(d_s, pi_old, pi)
        return loss - kl

    d_loss = jax.value_and_grad(loss_fn)

    # Has the FMAPG inner loop. 
    def _fn(pi, adv, d_s):
        pi_old = pi.copy()
        v_old = eval_pi(pi)
        for t in range(iterations): # m (number of inner loop iters is hard-coded)
            loss, grad = d_loss(pi, pi_old, d_s, adv)
            
            ### Need to check these
            pi = jnp.log(pi) + config.lr * grad 
            pi = jax.nn.softmax(pi)

        v_kp1 = eval_pi(pi)
        d_s = get_dpi(env, pi) 
        kl = kl_fn(d_s, pi, pi_old)
        stats = {
            "pi/return": (env.p0 @ v_kp1),
            "pi/improve": (env.p0 @ (v_kp1 - v_old)),
            "pi/kl_fwd": kl
        }
        return pi, v_kp1, stats

    return _fn

# this is the function called by main
def policy_iteration(env, pi_opt, eta, stop_criterion, policy=None): # prob shouldn't call it PI :)
    data = []
    value = get_value(env, policy) # get V^\pi for logging
    for step in itertools.count(1):

        # calls the corresponding method. How does this work? pi_opt has 2 components?
        policy, value, stats = pi_opt(env=env, pi_old=policy, eta=eta) 

        data.append((policy, value))
        if stop_criterion(step) or not is_prob_mass(policy):
            break
    pis, vs = list(map(lambda x: jnp.stack(x, 0), list(zip(*data)))) # form list for plotting?
    return policy, value, pis, vs
