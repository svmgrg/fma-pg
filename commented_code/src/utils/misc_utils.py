import numpy as np
from jax import numpy as jnp

import config
# from src.algorithms import v_iteration, q_iteration


def get_star(env):
    v_star = v_iteration(env)
    q_star = q_iteration(env)
    pi_star = get_pi_star(q_star)
    return pi_star, v_star


def get_pi_star(q):
    pi = np.zeros_like(q)
    idx = q.argmax(1)
    for i in range(pi.shape[0]):
        pi[i, idx[i]] = 1
    return pi


def is_prob_mass(pg_pi):
    return jnp.allclose(pg_pi.sum(1), 1) and (pg_pi.min() >= 0).all()


def save_stats(stats, global_step):
    for k, v in stats.items():
        config.tb.add_scalar(k, v, global_step)



### Ask about this. Might be helpful later
def line_search(pi, adv, eta_0, step_size):
    eta = eta_0
    while True:
        pi_new = softmax_ppo(pi, adv, eta)
        if not is_prob_mass(pi_new):
            return softmax_ppo(pi, adv, eta - step_size)
        eta = eta + step_size
