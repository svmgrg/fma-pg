import pdb
import time

from environments import CliffWorld
from utils import *

#======================================================================
# actual learning code
#======================================================================
gamma_list = [0.9]
num_steps = 200
eta_list = [0.03]
pg_method = 'sPPO'
num_inner_updates_list = [100, 50, 10]
alpha = 0.35
num_traj_est_adv_list = [100, 50, 10]

color_dict = {100: 'tab:red', 50: 'tab:green', 10: 'tab:blue'}
linestyle_dict = {100: '--', 50: '-.', 10: ':'}

adv_est_alg = 'sarsa'
adv_est_stepsize = 0.5

#======================================================================
# learning curves against the number of iterations (different steps)
#======================================================================
fig, axs = plt.subplots(1, 1)
tic = time.time()

for gamma in gamma_list:
    env = CliffWorld()
    v_star = np.dot(env.calc_v_star(), env.mu) # evaluate the optimal policy

    for eta in eta_list:
        # analytical FMA-PG
        vpi_analytical_dict = run_experiment_exact(
            num_iters=num_steps, gamma=gamma, eta=eta)
        plt.plot(vpi_analytical_dict[pg_method], label='analytical',
                 color='black')
        
        for num_inner_updates in num_inner_updates_list:
            # gradient based FMA-PG
            FLAG_USE_TRUE_ADVANTAGE = True
            vpi_list_inner, vpi_list_outer = run_experiment_approx(
                env=env, pg_method=pg_method, gamma=gamma, num_iters=num_steps,
                eta=eta, num_inner_updates=num_inner_updates, alpha=alpha,
                FLAG_USE_TRUE_ADVANTAGE=FLAG_USE_TRUE_ADVANTAGE,
                num_traj_est_adv=None)
            plt.plot(vpi_list_inner[:, -1], color=color_dict[num_inner_updates],
                     label='FMAPG_m:{}'.format(num_inner_updates))

            # gradient based FMA-PG with estimated advantage function
            FLAG_USE_TRUE_ADVANTAGE = False
            for num_traj_est_adv in num_traj_est_adv_list:
                vpi_list_inner, vpi_list_outer = run_experiment_approx(
                    env=env, pg_method=pg_method, gamma=gamma,
                    num_iters=num_steps, eta=eta,
                    num_inner_updates=num_inner_updates, alpha=alpha,
                    FLAG_USE_TRUE_ADVANTAGE=FLAG_USE_TRUE_ADVANTAGE,
                    num_traj_est_adv=num_traj_est_adv, adv_est_alg=adv_est_alg,
                    adv_est_stepsize=adv_est_stepsize)

                plt.plot(vpi_list_inner[:, -1],
                         color=color_dict[num_inner_updates],
                         linestyle=linestyle_dict[num_traj_est_adv],
                         label='FMAPG_m:{}_t:{}'.format(num_inner_updates,
                                                        num_traj_est_adv))
print('Total time taken: {}'.format(time.time() - tic))
axs.set_ylim([0, v_star])
plt.legend()
# plt.show()
plt.savefig('learning_curves_against_iters_{}_{}_{}.pdf'.format(
    pg_method, eta, adv_est_alg))
plt.close()

#======================================================================
# learning curves against the number of update steps
#======================================================================
# fig, axs = plt.subplots(1, 1)

# for gamma in gamma_list:
#     env = CliffWorld()
#     v_star = np.dot(env.calc_v_star(), env.mu) # evaluate the optimal policy

#     for eta in eta_list:
#         # analytical FMA-PG
#         vpi_analytical_dict = run_experiment_exact(
#             num_iters=num_steps, gamma=gamma, eta=eta)
#         plt.plot(vpi_analytical_dict[pg_method], label='analytical')
        
#         for num_inner_updates in num_inner_updates_list:
#             num_iters = int(num_steps / num_inner_updates)
            
#             # gradient based FMA-PG
#             vpi_list_inner, vpi_list_outer = run_experiment_approx(
#                 env=env, pg_method=pg_method, gamma=gamma,  num_iters=num_iters,
#                 eta=eta, num_inner_updates=num_inner_updates, alpha=alpha)
#             plt.plot(vpi_list_inner[:, 0:-1].flatten(), # vpi_list_inner[:, -1],
#                      label='FMAPG_m:{}'.format(num_inner_updates))
# axs.set_ylim([0, v_star + 0.1])
# plt.legend()
# plt.savefig('learning_curves_against_steps.pdf')
    
#----------------------------------------------------------------------
# testing code
#----------------------------------------------------------------------
# v_star = np.dot(env.calc_v_star(), env.mu)
# pi_star = env.calc_pi_star()
# v2 = env.calc_vpi(pi_star, FLAG_V_S0=True)

# fig, ax = plt.subplots(1, 1)
# plot_grid(ax, xlim=7, ylim=7)
# plot_policy(ax, pi_star, xlim=7, ylim=7)
# plt.axis('equal')
# plt.show()

# pdb.set_trace()
