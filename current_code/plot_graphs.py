import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import argparse

VSTAR = 0.9**6
color_dict = {'sPPO': 'blue', 'MDPO': 'red',
              'PPO': 'green', 'TRPO': 'tab:orange'}

def plot_alpha_sensitivity(ax, idx, num_outer_loops, num_inner_loops, pg_alg,
                           use_analytical_gradient, linewidth=1):

    if pg_alg in ['sPPO', 'MDPO']:
        big_list = [2**i for i in range(-13, 4, 1)] # eta
        if use_analytical_gradient: 
            small_list = [None] # alpha
        else:
            small_list = [2**i for i in range(-13, 4, 1)] # alpha
        delta = epsilon = decay_factor = None

        ax.plot(np.log(big_list), [VSTAR] * len(big_list), color='black',
                linestyle=':', linewidth=0.5)

        for alpha, tmp in zip(small_list, range(len(small_list))):
            c1 = np.array(mpl.colors.to_rgb(color_dict[pg_alg]))
            c2 = np.array(mpl.colors.to_rgb('white'))
            mix = (tmp + 1) / (len(small_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
            final_perf_list = []

            if use_analytical_gradient:
                plt_color = color_dict[pg_alg]

            float_alpha = float(alpha) if alpha is not None else None
                        
            for eta in big_list:
                filename='fmapg_data/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}__eta_{}__alpha_{}__epsilon_{}'\
                    '__delta_{}__decayFactor_{}__analyticalGrad_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops, eta,
                        float_alpha, epsilon, delta, decay_factor,
                        use_analytical_gradient)
                with open(filename, 'r') as fp:
                    dat = json.load(fp)

                final_perf_list.append(dat['vpi_outer_list'][idx])
                
            ax.plot(np.log(big_list), final_perf_list, color=plt_color,
                    label=r'$\alpha: {}$'.format(alpha), linewidth=linewidth)
    elif pg_alg == 'PPO':
        big_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # epsilon
        small_list = [2**i for i in range(-13, 4, 1)] # alpha
        eta = delta = decay_factor = None

        ax.plot(np.log(big_list), [VSTAR] * len(big_list), color='black',
                linestyle=':', linewidth=0.5)

        for alpha, tmp in zip(small_list, range(len(small_list))):
            c1 = np.array(mpl.colors.to_rgb(color_dict[pg_alg]))
            c2 = np.array(mpl.colors.to_rgb('white'))
            mix = (tmp + 1) / (len(small_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
            final_perf_list = []
            
            for epsilon in big_list:
                filename='fmapg_data/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}__eta_{}__alpha_{}__epsilon_{}'\
                    '__delta_{}__decayFactor_{}__analyticalGrad_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops, eta,
                        float(alpha), epsilon, delta, decay_factor,
                        use_analytical_gradient)
                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])
                
            ax.plot(np.log(big_list), final_perf_list, color=plt_color,
                    label=r'$\alpha: {}$'.format(alpha), linewidth=linewidth)
    elif pg_alg == 'TRPO':
        big_list = [2**i for i in range(-13, 14, 2)] # delta
        small_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # decay_fctr
        eta = alpha = epsilon = None

        ax.plot(np.log(big_list), [VSTAR] * len(big_list), color='black',
                linestyle=':', linewidth=0.5)

        for decay_factor, tmp in zip(small_list, range(len(small_list))):
            c1 = np.array(mpl.colors.to_rgb(color_dict[pg_alg]))
            c2 = np.array(mpl.colors.to_rgb('white'))
            mix = (tmp + 1) / (len(small_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
            final_perf_list = []
            
            for delta in big_list:
                filename='fmapg_data/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}__eta_{}__alpha_{}__epsilon_{}'\
                    '__delta_{}__decayFactor_{}__analyticalGrad_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops, eta,
                        alpha, epsilon, float(delta), decay_factor,
                        use_analytical_gradient)
                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])
                
            ax.plot(np.log(big_list), final_perf_list, color=plt_color,
                    label=r'decay_factor: {}$'.format(decay_factor),
                    linewidth=linewidth)
    else:
        raise NotImplementedError()

#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
fig, axs = plt.subplots(4, 5, figsize=(15, 10))
idx = -1

for row, pg_alg in zip(range(4), ['sPPO', 'MDPO', 'PPO', 'TRPO']):
    for col, num_inner_loops in zip(range(5), [1, 10, 100, 1000, None]):
        if col == 4 and pg_alg in ['sPPO', 'MDPO']:
            use_analytical_gradient = True
        else:
            use_analytical_gradient = False
            
        if ((col == 4 and pg_alg not in ['sPPO', 'MDPO'])
            or (col == 3 and pg_alg == 'TRPO')):
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].spines['bottom'].set_visible(False)
            axs[row, col].spines['left'].set_visible(False)
            axs[row, col].spines['right'].set_visible(False)
            axs[row, col].axes.xaxis.set_ticks([])
            axs[row, col].axes.yaxis.set_ticks([])
        else:
            plot_alpha_sensitivity(
                ax=axs[row, col], idx=idx, num_outer_loops=2000,
                num_inner_loops=num_inner_loops, pg_alg=pg_alg,
                use_analytical_gradient=use_analytical_gradient)
            axs[row, col].spines['right'].set_visible(False)
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].set_ylim([0, 0.6])
            # axs[row, col].legend()

# ax.set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
# ax.set_xlabel(r'$\log(\alpha)$')

plt.savefig('iter_{}__pg_sensitivity2.pdf'.format(idx))
plt.close()

#----------------------------------------------------------------------
# Find the best hyper-parameter configuration
#----------------------------------------------------------------------
# best_param_dict = dict()
# for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
#     best_param_dict[FLAG_PG_TYPE] = dict()
#     best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize'] = None
#     max_return_alpha = -1 * np.inf
    
#     for policy_stepsize in policy_stepsize_list:
#         best_param_dict[FLAG_PG_TYPE][policy_stepsize] = None
#         max_return_beta = -1 * np.inf
        
#         for critic_stepsize in critic_stepsize_list:
#             # read and process the data
#             dat = process_data(folder_name=folder_name,
#                                FLAG_PG_TYPE=FLAG_PG_TYPE,
#                                policy_stepsize=policy_stepsize,
#                                critic_stepsize=critic_stepsize,
#                                plotting_bin_size=plotting_bin_size)
#             final_perf_mean, _ = calc_final_perf(dat, idx=19)

#             if final_perf_mean > max_return_beta:
#                 max_return_beta = final_perf_mean
#                 best_param_dict[FLAG_PG_TYPE][policy_stepsize] = critic_stepsize
                        
#         if max_return_beta > max_return_alpha:
#             max_return_alpha = max_return_beta
#             best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize'] \
#                 = policy_stepsize

#----------------------------------------------------------------------
# Learning Curves
#----------------------------------------------------------------------
# num_fig_cols = 3 if FLAG_CAPTURE_ENTROPY else 2
# fig, axs = plt.subplots(num_fig_cols, 1, figsize=(5, 12), sharex=True)

# # stochastic PG
# for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
#     policy_stepsize = best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize']
#     print(FLAG_PG_TYPE, policy_stepsize)
#     critic_stepsize = best_param_dict[FLAG_PG_TYPE][policy_stepsize]
#     plt_color = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
#     # axs.set_title('{}_{}_{}'.format(
#     #     FLAG_PG_TYPE, policy_stepsize, critic_stepsize))
#     ax_ent = axs[2] if FLAG_CAPTURE_ENTROPY else None
#     plot_learning_curves(ax=axs[0], folder_name=folder_name,
#                          FLAG_PG_TYPE=FLAG_PG_TYPE,
#                          policy_stepsize=policy_stepsize,
#                          critic_stepsize=critic_stepsize,
#                          plt_color=plt_color,
#                          plotting_bin_size=plotting_bin_size,
#                          ax_v=axs[1], ax_ent=axs[2])

# axs[0].legend()
# for i in range(2):
#     axs[i].spines['right'].set_visible(False)
#     axs[i].spines['top'].set_visible(False)
# axs[1].set_xlabel('Timesteps')

# plt.show()
# plt.savefig('{}_learning_curves.pdf'.format(folder_name))
