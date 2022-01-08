import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import argparse
import copy



pg_method = 'TRPO'
folder_name = 'fmapg_DAT/CliffWorld_{}'.format(pg_method)
num_outer_loop = 2000
num_inner_loop = 10
FLAG_SAVE_INNER_STEPS = False
alpha_max = None
FLAG_WARM_START = False
warm_start_factor = None
max_backtracking_steps = None
optim_type = 'regularized'
stepsize_type = 'fixed'
eta_list = [float(2**i) for i in range(-13, 4)]
epsilon = None
delta = None
alpha_fixed = 0.5
decay_factor = None
armijo_const = None

final_perf_list = []
total_cnt_neg_list = []
percentage_cnt_neg_list = []
total_cnt_neg_adv_list = []
pi_list = []
fig, axs = plt.subplots(2, 1, figsize=(4, 6))
for eta in eta_list:
    filename='{}/nmOtrLp_{}__nmInrLp_{}'\
        '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
        '__mxBktStps_{}__optmTyp_{}__stpTyp_{}__eta_{}__eps_{}'\
        '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
            folder_name, num_outer_loop, num_inner_loop,
            FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
            warm_start_factor, max_backtracking_steps, optim_type,
            stepsize_type, eta, epsilon, delta, alpha_fixed,
            decay_factor, armijo_const)

    with open(filename, 'r') as fp:
        dat = json.load(fp)
    final_perf_list.append(dat['vpi_outer_list'][-1])

axs[0].plot((np.log(eta_list) / np.log(2)), final_perf_list, color='tab:blue')
axs[0].invert_xaxis()
axs[0].set_ylim([0, 0.6])
plt.show()
exit()

pg_method = 'sPPO'
folder_name = 'fmapg_DAT/CliffWorld_{}'.format(pg_method)
num_outer_loop = 2000
num_inner_loop = None
FLAG_SAVE_INNER_STEPS = False
alpha_max = None
FLAG_WARM_START = False
warm_start_factor = None
max_backtracking_steps = None
optim_type = 'analytical'
stepsize_type = 'fixed'
eta_list = [float(2**i) for i in range(-13, 10)]
epsilon = None
delta = None
alpha_fixed = None
decay_factor = None
armijo_const = None

final_perf_list = []
total_cnt_neg_list = []
percentage_cnt_neg_list = []
total_cnt_neg_adv_list = []
pi_list = []
fig, axs = plt.subplots(2, 1, figsize=(4, 6))
for eta in eta_list:
    filename='{}/nmOtrLp_{}__nmInrLp_{}'\
        '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
        '__mxBktStps_{}__optmTyp_{}__stpTyp_{}__eta_{}__eps_{}'\
        '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
            folder_name, num_outer_loop, num_inner_loop,
            FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
            warm_start_factor, max_backtracking_steps, optim_type,
            stepsize_type, eta, epsilon, delta, alpha_fixed,
            decay_factor, armijo_const)

    with open(filename, 'r') as fp:
        dat = json.load(fp)
    final_perf_list.append(dat['vpi_outer_list'][-1])
    total_cnt_neg_list.append(np.sum(dat['cnt_neg_list']) / (2000 * 21 * 4))
    total_cnt_neg_adv_list.append(
        np.sum(dat['cnt_neg_adv_list']) / (2000 * 21 * 4))
    percentage_cnt_neg_list.append(
        (np.array(dat['cnt_neg_list']) \
         / np.array(dat['cnt_neg_adv_list'])).sum() / 2000)
    pi_list.append(np.count_nonzero(np.array(dat['pi']) < 1e-3) / (21 * 4))

axs[0].plot((np.log(eta_list) / np.log(2)), final_perf_list, color='tab:blue')
axs[0].invert_xaxis()

axs[1].plot((np.log(eta_list) / np.log(2)), total_cnt_neg_list,
            color='tab:purple')
axs[1].plot((np.log(eta_list) / np.log(2)), total_cnt_neg_adv_list,
            color='tab:green')
axs[1].plot((np.log(eta_list) / np.log(2)), percentage_cnt_neg_list,
            color='black')
axs[1].plot((np.log(eta_list) / np.log(2)), pi_list, color='tab:red',
            linestyle='--')

axs[1].invert_xaxis()
plt.savefig('sPPO_analytical.pdf')
exit()















VSTAR = 0.9**6
color_dict = {'sPPO': 'blue', 'MDPO': 'red', 'PPO': 'green',
              'TRPO': 'tab:orange', 'TRPO_KL': 'tab:purple',
              'TRPO_KL_LS': 'cyan'}


def plot_alpha_sensitivity(ax, idx, num_outer_loops, num_inner_loops, pg_alg,
                           use_analytical_gradient, linewidth=1):
    eta = alpha = delta = epsilon = decay_factor = zeta \
        = armijo_constant = max_backtracking_iters = FLAG_BETA_WARM_START \
            = warm_start_beta_init = warm_start_beta_factor = None

    eta_list = alpha_list = [2**i for i in range(-13, 4, 1)]
    epsilon_list = decay_factor_list \
        = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    zeta_list = delta_list = [2**i for i in range(-13, 14, 2)]
    c1 = np.array(mpl.colors.to_rgb(color_dict[pg_alg]))
    c2 = np.array(mpl.colors.to_rgb('white'))

    if pg_alg in ['sPPO', 'MDPO']:
        FLAG_BETA_WARM_START = True
        
        if use_analytical_gradient: 
            alpha_list = [None] # alpha

        for alpha, tmp in zip(alpha_list, range(len(alpha_list))):
            if use_analytical_gradient:
                plt_color = color_dict[pg_alg]
            else:
                mix = (tmp + 1) / (len(alpha_list) + 2)
                plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

            final_perf_list = []
            float_alpha = float(alpha) if alpha is not None else None
            for eta in eta_list:
                filename='PG_expmts/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}'\
                    '__eta_{}__alpha_{}__epsilon_{}__delta_{}__decayFactor_{}'\
                    '__analyticalGrad_{}__zeta_{}__armijoConstant_{}'\
                    '__maxBacktrackingIters_{}__FlagBetaWarmStart_{}'\
                    '__wsbI_{}__wsbF_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops,
                        float(eta), float_alpha, epsilon, delta, decay_factor,
                        use_analytical_gradient, zeta, armijo_constant,
                        max_backtracking_iters, FLAG_BETA_WARM_START,
                        warm_start_beta_init, warm_start_beta_factor)

                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])
                
            ax.plot((np.log(eta_list) / np.log(2)), final_perf_list,
                    color=plt_color, label=r'$\alpha: {}$'.format(alpha),
                    linewidth=linewidth)
        ax.plot(np.log(eta_list) / np.log(2), [VSTAR] * len(eta_list),
            color='black', linestyle=':', linewidth=0.5)
        ax.invert_xaxis()
            
    elif pg_alg == 'PPO':
        FLAG_BETA_WARM_START = True
        
        for alpha, tmp in zip(alpha_list, range(len(alpha_list))):
            mix = (tmp + 1) / (len(alpha_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
            
            final_perf_list = []
            for epsilon in epsilon_list:
                filename='PG_expmts/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}'\
                    '__eta_{}__alpha_{}__epsilon_{}__delta_{}__decayFactor_{}'\
                    '__analyticalGrad_{}__zeta_{}__armijoConstant_{}'\
                    '__maxBacktrackingIters_{}__FlagBetaWarmStart_{}'\
                    '__wsbI_{}__wsbF_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops,
                        eta, float(alpha), epsilon, delta, decay_factor,
                        use_analytical_gradient, zeta, armijo_constant,
                        max_backtracking_iters, FLAG_BETA_WARM_START,
                        warm_start_beta_init, warm_start_beta_factor)

                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])

            ax.plot(epsilon_list, final_perf_list,
                    color=plt_color, label=r'$\epsilon: {}$'.format(alpha),
                    linewidth=linewidth)
        ax.plot(epsilon_list, [VSTAR] * len(epsilon_list),
            color='black', linestyle=':', linewidth=0.5)
        ax.invert_xaxis()
        
    elif pg_alg == 'TRPO':
        FLAG_BETA_WARM_START = True
        
        for decay_factor, tmp in zip(
                decay_factor_list, range(len(decay_factor_list))):
            mix = (tmp + 1) / (len(decay_factor_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
       
            final_perf_list = []
            for delta in delta_list:
                filename='PG_expmts/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}'\
                    '__eta_{}__alpha_{}__epsilon_{}__delta_{}__decayFactor_{}'\
                    '__analyticalGrad_{}__zeta_{}__armijoConstant_{}'\
                    '__maxBacktrackingIters_{}__FlagBetaWarmStart_{}'\
                    '__wsbI_{}__wsbF_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops,
                        eta, alpha, epsilon, float(delta), decay_factor,
                        use_analytical_gradient, zeta, armijo_constant,
                        max_backtracking_iters, FLAG_BETA_WARM_START,
                        warm_start_beta_init, warm_start_beta_factor)

                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])
                
            ax.plot(np.log(delta_list) / np.log(2),
                    final_perf_list, color=plt_color,
                    label=r'decay_factor: {}$'.format(decay_factor),
                    linewidth=linewidth)
        ax.plot(np.log(delta_list) / np.log(2), [VSTAR] * len(delta_list),
            color='black', linestyle=':', linewidth=0.5)

        ax.invert_xaxis()

    elif pg_alg == 'TRPO_KL':
        for alpha, tmp in zip(alpha_list, range(len(alpha_list))):
            
            
            mix = (tmp + 1) / (len(alpha_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
            
            FLAG_BETA_WARM_START = True
            
            final_perf_list = []
            for zeta in zeta_list:
                # zeta = -1 * zeta # to plot for -zeta; also use "__minusZeta"
                filename='PG_expmts/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}'\
                    '__eta_{}__alpha_{}__epsilon_{}__delta_{}__decayFactor_{}'\
                    '__analyticalGrad_{}__zeta_{}__armijoConstant_{}'\
                    '__maxBacktrackingIters_{}__FlagBetaWarmStart_{}'\
                    '__wsbI_{}__wsbF_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops,
                        eta, float(alpha), epsilon, delta, decay_factor,
                        use_analytical_gradient, float(zeta), armijo_constant,
                        max_backtracking_iters, FLAG_BETA_WARM_START,
                        warm_start_beta_init, warm_start_beta_factor)

                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])

            ax.plot(np.log(zeta_list) / np.log(2), final_perf_list,
                    color=plt_color, label=r'$\alpha: {}$'.format(alpha),
                    linewidth=linewidth)
        ax.plot(np.log(zeta_list) / np.log(2), [VSTAR] * len(zeta_list),
            color='black', linestyle=':', linewidth=0.5)

    elif pg_alg == 'TRPO_KL_LS':
        for decay_factor, tmp in zip(
                decay_factor_list, range(len(decay_factor_list))):
            mix = (tmp + 1) / (len(decay_factor_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

            armijo_constant = 0
            FLAG_BETA_WARM_START = False
            warm_start_beta_init = 100
            warm_start_beta_factor = 1

            final_perf_list = []
            for zeta in zeta_list:
                # zeta = -1 * zeta
                filename='PG_expmts/CliffWorld_{}/numOuterLoops_{}'\
                    '__numInnerLoops_{}'\
                    '__eta_{}__alpha_{}__epsilon_{}__delta_{}__decayFactor_{}'\
                    '__analyticalGrad_{}__zeta_{}__armijoConstant_{}'\
                    '__maxBacktrackingIters_{}__FlagBetaWarmStart_{}'\
                    '__wsbI_{}__wsbF_{}'.format(
                        pg_alg, num_outer_loops, num_inner_loops,
                        eta, alpha, epsilon, delta, decay_factor,
                        use_analytical_gradient, float(zeta),
                        float(armijo_constant),
                        max_backtracking_iters, FLAG_BETA_WARM_START,
                        float(warm_start_beta_init),
                        float(warm_start_beta_factor))

                with open(filename, 'r') as fp:
                    dat = json.load(fp)
                final_perf_list.append(dat['vpi_outer_list'][idx])

            ax.plot(np.log(zeta_list) / np.log(2), final_perf_list,
                    color=plt_color,
                    label=r'decay_factor: {}$'.format(decay_factor),
                    linewidth=linewidth)
        ax.plot(np.log(zeta_list) / np.log(2), [VSTAR] * len(zeta_list),
            color='black', linestyle=':', linewidth=0.5)

    else:
        raise NotImplementedError()

#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
fig, axs = plt.subplots(6, 5, figsize=(15, 12))
idx = -1

alg_list = ['sPPO', 'MDPO', 'PPO', 'TRPO_KL', 'TRPO', 'TRPO_KL_LS']

for row, pg_alg in zip(range(len(alg_list)), alg_list):
    for col, num_inner_loops in zip(range(5), [1, 10, 100, 1000, None]):
        if col == 4 and pg_alg == 'MDPO':
            continue
        if col == 3 and (pg_alg == 'TRPO' or pg_alg == 'TRPO_KL_LS'):
            continue
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

plt.savefig('iter_{}__pg_sensitivity_new_minusZetaKL.pdf'.format(idx))
# plt.show()
plt.close()
exit()
#----------------------------------------------------------------------
# Find the best hyper-parameter configuration (not working right now!)
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
#                 best_param_dict[FLAG_PG_TYPE][policy_stepsize] \
#                     = critic_stepsize
                        
#         if max_return_beta > max_return_alpha:
#             max_return_alpha = max_return_beta
#             best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize'] \
#                 = policy_stepsize
           
#----------------------------------------------------------------------
# find the best hyperparameter configuration (lazy version :P)
#----------------------------------------------------------------------
pg_alg = 'TRPO_KL_LS'
num_outer_loops = 2000
num_inner_loops = 1000
alpha = decay_factor = epsilon = delta = eta = zeta = None
use_analytical_gradient = False
decay_factor_list = epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
eta_list = alpha_list = [2**i for i in range(-13, 4, 1)]
zeta_list = delta_list = [2**i for i in range(-13, +14, 2)]
max_ret = -np.inf
best_decay_factor = best_delta = best_epsilon = best_eta = best_alpha = None
for zeta in zeta_list:
    for delta in delta_list:
        float_alpha = float(alpha) if alpha is not None else None
        float_delta = float(delta) if delta is not None else None
        float_zeta = float(zeta) if zeta is not None else None
        filename='fmapg_data/CliffWorld_{}/numOuterLoops_{}'\
            '__numInnerLoops_{}__eta_{}__alpha_{}__epsilon_{}'\
            '__delta_{}__decayFactor_{}__analyticalGrad_{}__zeta_{}'.format(
                pg_alg, num_outer_loops, num_inner_loops, eta,
                float_alpha, epsilon, float_delta, decay_factor,
                use_analytical_gradient, float_zeta)
        with open(filename, 'r') as fp:
            dat = json.load(fp)
        if dat['vpi_outer_list'][-1] > max_ret:
            max_ret = dat['vpi_outer_list'][-1] 
            best_alpha = alpha
            best_epsilon = epsilon
            best_delta = delta
            best_decay_factor = decay_factor
            best_eta = eta
            best_zeta = zeta
        print(zeta, delta, dat['vpi_outer_list'][-1])
        print('best', best_zeta, best_delta, max_ret)
exit()

#----------------------------------------------------------------------
# Learning Curves
#----------------------------------------------------------------------
folder_name = 'fmapg_data'
num_outer_loops = 2000
use_analytical_gradient = False
best_param = dict()
best_param[100] = {
    'sPPO': {'eta':0.03125, 'alpha':0.25, 'eps':None, 'delta':None, 'df':None},
    'MDPO':{'eta':0.03125, 'alpha':1., 'eps':None, 'delta':None, 'df':None},
    'PPO':{'eta':None, 'alpha':8., 'eps':0.1, 'delta':None, 'df':None},
    'TRPO':{'eta':None, 'alpha':None, 'eps':None, 'delta':0.5, 'df':0.9},
    'TRPO_KL':{'eta':None, 'alpha':1., 'eps':None, 'delta':None, 'df':None,
               'zeta':32.}}

best_param[1000] = copy.deepcopy(best_param[100]) # almost same as before
best_param[1000]['MDPO']['eta'] = 0.0625; best_param[1000]['MDPO']['alpha'] = 2.
best_param[1000]['TRPO_KL']['zeta'] = 8.
best_param[1000]['TRPO_KL']['alpha'] = 2.

best_param[10] = copy.deepcopy(best_param[100]) # quite different
best_param[10]['sPPO']['eta'] = 0.125; best_param[10]['sPPO']['alpha'] = 4.
best_param[10]['MDPO']['eta'] = 0.125; best_param[10]['MDPO']['alpha'] = 4.
best_param[10]['PPO']['alpha'] = 8.; best_param[10]['PPO']['eps'] = 0.6
best_param[10]['TRPO_KL']['zeta'] = 8.
best_param[10]['TRPO_KL']['alpha'] = 4.


fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True)
for col, m in zip(range(3), [10, 100, 1000]):
    for pg_method in ['sPPO', 'MDPO', 'PPO', 'TRPO', 'TRPO_KL']:
        num_inner_loops = 100 if (pg_method == 'TRPO' and m == 1000) else m
        eta = best_param[m][pg_method]['eta']
        alpha = best_param[m][pg_method]['alpha']
        epsilon = best_param[m][pg_method]['eps']
        delta = best_param[m][pg_method]['delta']
        decay_factor = best_param[m][pg_method]['df']
        zeta = None

        if pg_method == 'TRPO_KL':
            zeta = best_param[m][pg_method]['zeta']
            filename='{}/CliffWorld_{}/numOuterLoops_{}'\
                '__numInnerLoops_{}__eta_{}__alpha_{}__epsilon_{}__delta_{}'\
                '__decayFactor_{}__analyticalGrad_{}__zeta_{}'.format(
                    folder_name, pg_method, num_outer_loops, num_inner_loops,
                    eta, alpha, epsilon, delta, decay_factor,
                    use_analytical_gradient, zeta)
        else:
            filename='{}/CliffWorld_{}/numOuterLoops_{}'\
                '__numInnerLoops_{}__eta_{}__alpha_{}__epsilon_{}'\
                '__delta_{}__decayFactor_{}__analyticalGrad_{}'.format(
                    folder_name, pg_method, num_outer_loops, num_inner_loops,
                    eta, alpha, epsilon, delta, decay_factor,
                    use_analytical_gradient)
        with open(filename, 'r') as fp:
            dat = json.load(fp)

        axs[col].plot(dat['vpi_outer_list'], color=color_dict[pg_method],
                      label='{}_{}_{}_{}_{}_{}_{}'.format(
                          pg_method, eta, alpha, epsilon, delta,
                          decay_factor, zeta),
                      linewidth=1)
    axs[col].legend()
    axs[col].set_ylim([0, 0.6])
    axs[col].spines['right'].set_visible(False)
    axs[col].spines['top'].set_visible(False)
plt.savefig('{}_learning_curves.pdf'.format(folder_name))
plt.close()

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
