import pdb
import time

from environments import *
from utils import *

#----------------------------------------------------------------------
# actual learning code
#----------------------------------------------------------------------
gamma = 0.9
episode_cutoff_length = 1000
reward_noise = 0

env = CliffWorld(
    P=P, r=r, mu=mu, terminal_states=terminal_states,
    gamma=gamma, episode_cutoff_length=episode_cutoff_length,
    reward_noise=reward_noise)

pg_method = 'TRPO'
num_iters = 10
eta = None # 0.5
delta = 1
epsilon = None # 0.1
decay_factor = 0.9

FLAG_ANALYTICAL_GRADIENT = False
num_inner_updates = 1
alpha = None # 0.35

FLAG_TRUE_ADVANTAGE = True
num_traj_estimate_adv = None 
adv_estimate_alg = None
adv_estimate_stepsize = None

FLAG_SAVE_INNER_STEPS = False

#----------------------------------------------------------------------
# learning curves against the number of outer loop iterations
#----------------------------------------------------------------------
tic = time.time()
vpi_list_outer, vpi_list_inner, pi = run_experiment(
    env=env, pg_method=pg_method, num_iters=num_iters,
    eta=eta, delta=delta, decay_factor=decay_factor, epsilon=epsilon,
    FLAG_ANALYTICAL_GRADIENT=FLAG_ANALYTICAL_GRADIENT,
    num_inner_updates=num_inner_updates, alpha=alpha,
    FLAG_TRUE_ADVANTAGE=FLAG_TRUE_ADVANTAGE,
    adv_estimate_alg=adv_estimate_alg,
    num_traj_estimate_adv=num_traj_estimate_adv,
    adv_estimate_stepsize=adv_estimate_stepsize,
    FLAG_SAVE_INNER_STEPS=FLAG_SAVE_INNER_STEPS)
print('Total time taken: {}'.format(time.time() - tic))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

plot_grid(axs[0])
plot_policy(axs[0], pi)

axs[1].plot(vpi_list_outer)
            
v_star = np.dot(env.calc_v_star(), env.mu)
axs[1].set_ylim([-1, v_star + 0.05])

# plt.savefig('lc_{}_{}_env.pdf'.format(
#     pg_method, eta))
plt.show()  
plt.close()
