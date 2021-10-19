import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

from utils import *
from environments import *

# v_star = np.dot(env.calc_v_star(), env.mu)
# pi_star = env.calc_pi_star()
# v2 = env.calc_vpi(pi_star, FLAG_V_S0=True)

# fig, ax = plt.subplots(1, 1)
# plot_grid(ax, xlim=7, ylim=7)
# plot_policy(ax, pi_star, xlim=7, ylim=7)
# plt.axis('equal')
# plt.show()

#======================================================================
# Testing code (plotting dynamics)
#======================================================================
env_simone = CliffWorld(
    P=P_simone, r=r_simone, mu=mu_simone, terminal_states=terminal_states_simone,
    gamma=0.9, episode_cutoff_length=1000, reward_noise=0)

env = CliffWorld(
    P=P, r=r, mu=mu, terminal_states=terminal_states,
    gamma=0.9, episode_cutoff_length=1000, reward_noise=0)

color_list = ['tab:orange', 'tab:blue', 'tab:red', 'tab:green']
x_diff = [0.3, 0.7, 0.5, 0.5]
y_diff = [0.5, 0.5, 0.3, 0.7]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

xlim = ylim = 7
# make grid
for x in range(xlim + 1):
    axs[0, 0].plot((x, x), (0, ylim), linewidth=0.5, color='black')
for y in range(ylim + 1):
    axs[0, 0].plot((0, xlim), (y, y), linewidth=0.5, color='black')
# show the transition dynamics
for state_idx in range(xlim * ylim):
    x_old, y_old = state_idx // ylim, state_idx % ylim
    for action_idx in range(4):
        next_state_idx = env_simone.P[state_idx, action_idx].nonzero()[0].item()
        x_new, y_new = next_state_idx // ylim, next_state_idx % ylim
        axs[0, 0].quiver(x_old + x_diff[action_idx], y_old + y_diff[action_idx],
                      x_new - x_old, y_new - y_old,
                      color=color_list[action_idx], width=0.007, headwidth=2,
                      headlength=4, scale=1, scale_units='xy', linewidth=0.1)
# show the state indices
for x in range(xlim):
    for y in range(ylim):
        state_idx = ylim * x + y
        axs[0, 0].text(x + 0.3, y + 0.3, str(state_idx),
                    fontsize='large', fontweight='bold')

xlim = 4
ylim = 5
# make grid
for x in range(xlim + 2):
    axs[0, 1].plot((x, x), (0, ylim), linewidth=0.5, color='black')
for y in range(ylim + 1):
    axs[0, 1].plot((0, xlim + 1), (y, y), linewidth=0.5, color='black')
# show the transition dynamics
for state_idx in range(xlim * ylim + 1):
    x_old, y_old = state_idx // ylim, state_idx % ylim
    for action_idx in range(4):
        next_state_idx = env.P[state_idx, action_idx].nonzero()[0].item()
        x_new, y_new = next_state_idx // ylim, next_state_idx % ylim
        axs[0, 1].quiver(x_old + x_diff[action_idx], y_old + y_diff[action_idx],
                      x_new - x_old, y_new - y_old,
                      color=color_list[action_idx], width=0.007, headwidth=2,
                      headlength=4, scale=1, scale_units='xy', linewidth=0.1)
# show the state indices
for state_idx in range(21):
    x = state_idx // ylim
    y = state_idx % ylim
    axs[0, 1].text(x + 0.3, y + 0.3, str(state_idx),
                fontsize='large', fontweight='bold')

#======================================================================
# Testing code (testing similarity of learning curves from the paper)
#======================================================================
pg_method = 'MDPO'
num_iters = 2000
eta = 0.03

FLAG_ANALYTICAL_GRADIENT = True
num_inner_updates = None # 100
alpha = None # 0.1

FLAG_TRUE_ADVANTAGE = True
num_traj_estimate_adv = None # 100
adv_estimate_alg = None # 'sarsa'
adv_estimate_stepsize = None # 0.5

FLAG_SAVE_INNER_STEPS = False

#----------------------------------------------------------------------
# learning curves against the number of outer loop iterations
#----------------------------------------------------------------------
tic = time.time()
for env_try, col, env_name in \
    zip([env_simone, env], range(2), ['simone', 'smaller']):
    for pg_method in ['MDPO', 'sPPO']:
        for eta in [0.03, 1]:
            vpi_list_outer, vpi_list_inner = run_experiment(
                env=env_try, pg_method=pg_method, num_iters=num_iters, eta=eta,
                FLAG_ANALYTICAL_GRADIENT=FLAG_ANALYTICAL_GRADIENT,
                num_inner_updates=num_inner_updates, alpha=alpha,
                FLAG_TRUE_ADVANTAGE=FLAG_TRUE_ADVANTAGE,
                adv_estimate_alg=adv_estimate_alg,
                num_traj_estimate_adv=num_traj_estimate_adv,
                adv_estimate_stepsize=adv_estimate_stepsize,
                FLAG_SAVE_INNER_STEPS=FLAG_SAVE_INNER_STEPS)
            
            axs[1, col].plot(vpi_list_outer, label='{}_{}_{}'.format(
                pg_method, eta, env_name))
            v_star = np.dot(env_try.calc_v_star(), env_try.mu)
            axs[1, col].set_ylim([0, v_star + 0.05])
            axs[1, col].legend()
    print('Total time taken: {}'.format(time.time() - tic))

plt.savefig('test_env.pdf')
plt.close()
