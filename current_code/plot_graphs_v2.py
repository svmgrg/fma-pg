import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

VSTAR = 0.9**6
color_dict = {'sPPO': 'blue', 'MDPO': 'red', 'PPO': 'green',
              'TRPO': 'tab:purple', 'TRPO_KL': 'tab:purple',
              'TRPO_KL_LS': 'cyan'}
linewidth = 1

num_inner_loop_list = [1, 10, 100, 1000]

eta_list = [2**i for i in range(-13, -2, 1)]
delta_list = [2**i for i in range(-13, 14, 2)]

alpha_list = [2**i for i in range(-13, 4, 1)]
armijo_const_list = [0.0, 0.001, 0.01, 0.1, 0.5]

pg_method = 'MDPO'
PLOT_TYPE = 'vpi_outer'

num_outer_loop = 2000
FLAG_SAVE_INNER_STEPS = False
alpha_max = None
FLAG_WARM_START = False
warm_start_factor = None
max_backtracking_steps = None
optim_type = 'regularized'
stepsize_type = 'fixed'
epsilon = None
alpha_fixed = None
decay_factor = None

folder_name = 'fmapg_DAT/CliffWorld_{}_{}_{}'.format(
    pg_method, optim_type, stepsize_type)
delta = armijo_const = None

num_cols = len(num_inner_loop_list)

#======================================================================
# Plot learning curves
#======================================================================
# num_rows = len(eta_list)
# fig, axs = plt.subplots(num_rows, num_cols, sharey=True.
#                         figsize=(3 * num_cols, 1.7 * num_rows))
# final_perf_idx = -1
# for col, num_inner_loop in enumerate(num_inner_loop_list):
#     for row, eta in enumerate(eta_list):
#         plot_sensitivity(ax=axs[row][col], num_inner_loop=num_inner_loop,
#                          eta=float(eta), pg_method=pg_method,
#                          final_perf_idx=final_perf_idx,
#                          plot_type=PLOT_TYPE, linewidth=linewidth)
#         axs[row][col].spines['right'].set_visible(False)
#         axs[row][col].spines['top'].set_visible(False)

#         if col == 0:
#             axs[row][col].set_ylabel(r'$\eta=' + str(eta) + '$')

#     axs[0][col].set_title('m={}'.format(num_inner_loop))

# axs[-1][3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#                   fancybox=True, shadow=True, ncol=6)

# plt.savefig('{}_{}_{}_{}.pdf'.format(
#     pg_method, optim_type, stepsize_type, PLOT_TYPE))
# exit()

def plot_sensitivity(ax, num_inner_loop, pg_method, final_perf_idx=-1, 
                     plot_type=None, linewidth=1):
    assert plot_type in ['vpi_outer', 'grad_lpi_inner', 'grad_jpi_outer']
    
    c1 = np.array(mpl.colors.to_rgb(color_dict[pg_method]))
    c2 = np.array(mpl.colors.to_rgb('white'))

    outer_list = alpha_list if stepsize_type == 'fixed' else armijo_const_list
    for outer_idx, clr_mix in zip(outer_list, range(len(outer_list))):
        alpha_fixed = float(outer_idx) if stepsize_type == 'fixed' else None
        armijo_const = float(outer_idx) \
            if stepsize_type == 'line_search' else None

        mix = (clr_mix + 1) / (len(outer_list) + 2)
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

        final_perf_list = []

        inner_list = eta_list if optim_type == 'regularized' else delta_list
        for inner_idx in inner_list:
            eta = float(inner_idx) if optim_type == 'regularized' else None
            delta = float(inner_idx) if optim_type == 'constrained' else None
                            
            filename='{}/nmOtrLp_{}__nmInrLp_{}'\
                '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
                '__mxBktStps_{}__eta_{}__eps_{}'\
                '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                    folder_name, num_outer_loop, num_inner_loop,
                    FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                    warm_start_factor, max_backtracking_steps, eta, epsilon,
                    delta, alpha_fixed, decay_factor, armijo_const)
            with open(filename, 'r') as fp:
                dat = json.load(fp)

            final_perf_list.append(dat[plot_type + '_list'][final_perf_idx])

        ax.plot(np.log(eta_list) / np.log(2), final_perf_list,
                color=plt_color, label=r'$\alpha:{}$'.format(alpha_fixed),
                linewidth=linewidth)
        
        if plot_type == 'vpi_outer':
            ax.plot(np.log(eta_list) / np.log(2), [VSTAR] * len(eta_list),
                    color='black', linestyle=':', linewidth=0.5)

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                        box.width, box.height * 0.9])
    ax.invert_xaxis()

#======================================================================
# Plot sensitivity plots
#======================================================================
num_cols = len(num_inner_loop_list)
fig, axs = plt.subplots(1, num_cols, figsize=(3 * num_cols, 1.7))
final_perf_idx = -1
for col, num_inner_loop in enumerate(num_inner_loop_list):
    plot_sensitivity(ax=axs[col], num_inner_loop=num_inner_loop,
                     pg_method=pg_method, final_perf_idx=final_perf_idx,
                     plot_type=PLOT_TYPE, linewidth=linewidth)
    axs[col].spines['right'].set_visible(False)
    axs[col].spines['top'].set_visible(False)
    if PLOT_TYPE == 'vpi_outer':
        axs[col].set_ylim([0, 0.6])
    elif PLOT_TYPE == 'grad':
        axs[col].set_ylim([0, 30])
    axs[col].set_title('m={}'.format(num_inner_loop))

    if col == 2:
        axs[col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
                        fancybox=True, shadow=True, ncol=6)
axs[0].set_xlabel(r'$\log_2(\eta)$')

# plt.show()
plt.savefig('{}_{}_{}_{}_sensitivity.pdf'.format(
    pg_method, optim_type, stepsize_type, PLOT_TYPE))
exit()



#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================




num_rows = len(alpha_list)
fig, axs = plt.subplots(num_rows, 3, figsize=(6 * 3, 4 * num_rows),
                        sharex=True)
for row, alpha_fixed in enumerate(alpha_list):
    for num_inner_loop in num_inner_loop_list:
        min_grad = +np.inf
        max_J = -np.inf
        chosen_eta = None
        for eta in eta_list:
            filename='{}/nmOtrLp_{}__nmInrLp_{}'\
                '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
                '__mxBktStps_{}__eta_{}__eps_{}'\
                '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                    folder_name, num_outer_loop, num_inner_loop,
                    FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                    warm_start_factor, max_backtracking_steps, eta, epsilon,
                    delta, alpha_fixed, decay_factor, armijo_const)

            with open(filename, 'r') as fp:
                dat = json.load(fp)

            # if max_J < dat['vpi_outer_list'][-1]:
            #     max_J = dat['vpi_outer_list'][-1]
            #     chosen_eta = eta
            if min_grad > dat['grad_pi_norm_list'][-1]:
                min_grad = dat['grad_pi_norm_list'][-1]
                chosen_eta = eta

        eta = chosen_eta
        filename='{}/nmOtrLp_{}__nmInrLp_{}'\
            '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
            '__mxBktStps_{}__eta_{}__eps_{}'\
            '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                folder_name, num_outer_loop, num_inner_loop,
                FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                warm_start_factor, max_backtracking_steps, eta, epsilon,
                delta, alpha_fixed, decay_factor, armijo_const)
        with open(filename, 'r') as fp:
            dat = json.load(fp)

        plt_label = r'$\eta=2^{}$'.format(
            int(np.log(eta) / np.log(2)))

        axs[row][0].plot(dat['vpi_outer_list'][10:], label=plt_label)
        axs[row][1].plot(dat['grad_norm_list'][10:], label=plt_label)
        axs[row][2].plot(dat['grad_pi_norm_list'][10:], label=plt_label)

        axs[row][0].legend()

        axs[row][0].set_ylim([0, 0.6])
        axs[row][0].set_ylabel('alpha=2^{}'.format(
            int(np.log(alpha_fixed) / np.log(2))))

        axs[row][0].set_title('J | {}'.format(pg_method))
        axs[row][1].set_title(r'$\|\nabla_{\theta(s, a)} \mathcal{J}\|$')
        axs[row][2].set_title(r'$\|\nabla_{\pi(a | s)} \mathcal{J}\|$')   

plt.savefig('{}__alpha_best_eta_min__grad_pi.png'.format(pg_method), dpi=150)
exit()


#======================================================================
num_rows = len(eta_list)
fig, axs = plt.subplots(num_rows, 3, figsize=(6 * 3, 4 * num_rows),
                        sharex=True)

for row, eta in enumerate(eta_list):
    for num_inner_loop in num_inner_loop_list:
        min_grad = +np.inf
        max_J = -np.inf
        chosen_alpha = None
        for alpha_fixed in alpha_list:
            filename='{}/nmOtrLp_{}__nmInrLp_{}'\
                '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
                '__mxBktStps_{}__eta_{}__eps_{}'\
                '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                    folder_name, num_outer_loop, num_inner_loop,
                    FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                    warm_start_factor, max_backtracking_steps, eta, epsilon,
                    delta, alpha_fixed, decay_factor, armijo_const)

            with open(filename, 'r') as fp:
                dat = json.load(fp)

            # if max_J < dat['vpi_outer_list'][-1]:
            #     max_J = dat['vpi_outer_list'][-1]
            #     chosen_alpha = alpha_fixed
            if min_grad > dat['grad_norm_list'][-1]:
                min_grad = dat['grad_norm_list'][-1]
                chosen_alpha = alpha_fixed

        alpha_fixed = chosen_alpha
        filename='{}/nmOtrLp_{}__nmInrLp_{}'\
            '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
            '__mxBktStps_{}__eta_{}__eps_{}'\
            '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                folder_name, num_outer_loop, num_inner_loop,
                FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                warm_start_factor, max_backtracking_steps, eta, epsilon,
                delta, alpha_fixed, decay_factor, armijo_const)
        with open(filename, 'r') as fp:
            dat = json.load(fp)

        plt_label = r'$\alpha=2^{}$'.format(
            int(np.log(alpha_fixed) / np.log(2)))

        axs[row][0].plot(dat['vpi_outer_list'][10:], label=plt_label)
        axs[row][1].plot(dat['grad_norm_list'][10:], label=plt_label)
        axs[row][2].plot(dat['grad_pi_norm_list'][10:], label=plt_label)

        axs[row][0].legend()

        axs[row][0].set_ylim([0, 0.6])
        axs[row][0].set_ylabel('eta=2^{}'.format(int(np.log(eta) / np.log(2))))

        axs[row][0].set_title('J | {}'.format(pg_method))
        axs[row][1].set_title(r'$\|\nabla_{\theta(s, a)} \mathcal{J}\|$')
        axs[row][2].set_title(r'$\|\nabla_{\pi(a | s)} \mathcal{J}\|$')   

plt.savefig('{}__eta_best_alpha.png'.format(pg_method), dpi=150)
exit()


#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================
#======================================================================





















VSTAR = 0.9**6
color_dict = {'sPPO': 'blue', 'MDPO': 'red', 'PPO': 'green',
              'TRPO': 'tab:purple', 'TRPO_KL': 'tab:purple',
              'TRPO_KL_LS': 'cyan'}
linewidth = 1

num_inner_loop_list = [1, 10, 100, 1000]

eta_list = [2**i for i in range(-13, 4, 1)]
delta_list = [2**i for i in range(-13, 14, 2)]

alpha_list = [2**i for i in range(-13, 4, 2)] # + [8192, 32768, 131072, 524288]
armijo_const_list = [0.0, 0.001, 0.01, 0.1, 0.5]

pg_method = 'sPPO'

num_outer_loop = 2000
FLAG_SAVE_INNER_STEPS = False
alpha_max = None
FLAG_WARM_START = False
warm_start_factor = None
max_backtracking_steps = None
optim_type = 'regularized'
stepsize_type = 'fixed'
epsilon = None
alpha_fixed = None
decay_factor = None

PLOT_TYPE = 'grad_norm'

folder_name = 'fmapg_DAT/CliffWorld_{}_{}_{}'.format(
    pg_method, optim_type, stepsize_type)

def plot_sensitivity(ax, num_inner_loop, pg_method, final_perf_idx=-1, 
                     plot_type=None, linewidth=1):
    assert plot_type in ['vpi', 'grad_norm', 'grad_pi_norm']
    
    c1 = np.array(mpl.colors.to_rgb(color_dict[pg_method]))
    c2 = np.array(mpl.colors.to_rgb('white'))

    outer_list = alpha_list if stepsize_type == 'fixed' else armijo_const_list
    for outer_idx, clr_mix in zip(outer_list, range(len(outer_list))):
        alpha_fixed = float(outer_idx) if stepsize_type == 'fixed' else None
        armijo_const = float(outer_idx) \
            if stepsize_type == 'line_search' else None

        mix = (clr_mix + 1) / (len(outer_list) + 2)
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

        final_perf_list = []

        inner_list = eta_list if optim_type == 'regularized' else delta_list
        for inner_idx in inner_list:
            eta = float(inner_idx) if optim_type == 'regularized' else None
            delta = float(inner_idx) if optim_type == 'constrained' else None
                            
            filename='{}/nmOtrLp_{}__nmInrLp_{}'\
                '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
                '__mxBktStps_{}__eta_{}__eps_{}'\
                '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                    folder_name, num_outer_loop, num_inner_loop,
                    FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                    warm_start_factor, max_backtracking_steps, eta, epsilon,
                    delta, alpha_fixed, decay_factor, armijo_const)

            with open(filename, 'r') as fp:
                dat = json.load(fp)

            if plot_type == 'vpi':
                final_perf_list.append(dat['vpi_outer_list'][final_perf_idx])
            elif plot_type == 'grad_norm':
                final_perf_list.append(dat['grad_norm_list'][final_perf_idx])
            elif plot_type == 'grad_pi_norm':
                final_perf_list.append(dat['grad_pi_norm_list'][final_perf_idx])

        ax.plot((np.log(eta_list) / np.log(2)), final_perf_list,
                     color=plt_color, label=r'$\alpha:{}$'.format(alpha_fixed),
                     linewidth=linewidth)
        if plot_type == 'vpi':
            ax.plot(np.log(eta_list) / np.log(2), [VSTAR] * len(eta_list),
                    color='black', linestyle=':', linewidth=0.5)
        ax.invert_xaxis()
    
num_cols = len(num_inner_loop_list)
fig, axs = plt.subplots(1, num_cols, figsize=(3 * num_cols, 1.7))
final_perf_idx = -1
for col, num_inner_loop in enumerate(num_inner_loop_list):
    plot_sensitivity(ax=axs[col], num_inner_loop=num_inner_loop,
                     pg_method=pg_method, final_perf_idx=final_perf_idx,
                     plot_type=PLOT_TYPE, linewidth=linewidth)
    axs[col].spines['right'].set_visible(False)
    axs[col].spines['top'].set_visible(False)
    if PLOT_TYPE == 'vpi':
        axs[col].set_ylim([0, 0.6])
    elif PLOT_TYPE == 'grad_pi_norm':
        axs[col].set_ylim([0, 30])
    axs[col].set_title('m={}'.format(num_inner_loop))

plt.show()
plt.savefig('{}_{}_{}_{}.pdf'.format(
    pg_method, optim_type, stepsize_type, PLOT_TYPE))
exit()

pg_method = 'MDPO'
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

folder_name = 'fmapg_DAT/CliffWorld_{}_{}_{}'.format(
    pg_method, optim_type, stepsize_type)

final_perf_list = []
total_cnt_neg_list = []
percentage_cnt_neg_list = []
total_cnt_neg_adv_list = []
pi_list = []
fig, axs = plt.subplots(2, 1, figsize=(4, 6))
for eta in eta_list:
    filename='{}/nmOtrLp_{}__nmInrLp_{}'\
        '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
        '__mxBktStps_{}__eta_{}__eps_{}'\
        '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
            folder_name, num_outer_loop, num_inner_loop,
            FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
            warm_start_factor, max_backtracking_steps, eta, epsilon,
            delta, alpha_fixed, decay_factor, armijo_const)

    with open(filename, 'r') as fp:
        dat = json.load(fp)
    final_perf_list.append(dat['vpi_outer_list'][-1])
    total_cnt_neg_list.append(np.sum(dat['cnt_neg_list']) / (2000 * 21 * 4))
    total_cnt_neg_adv_list.append(
        np.sum(dat['cnt_neg_adv_list']) / (2000 * 21 * 4))
    percentage_cnt_neg_list.append(
        (np.array(dat['cnt_neg_list']) \
         / np.array(dat['cnt_neg_adv_list'])).sum() / 2000)
    # pi_list.append(np.count_nonzero(np.array(dat['pi']) < 1e-3) / (21 * 4))

axs[0].plot((np.log(eta_list) / np.log(2)), final_perf_list, color='tab:blue')
axs[0].invert_xaxis()

axs[1].plot((np.log(eta_list) / np.log(2)), total_cnt_neg_list,
            color='tab:purple')
axs[1].plot((np.log(eta_list) / np.log(2)), total_cnt_neg_adv_list,
            color='tab:green')
axs[1].plot((np.log(eta_list) / np.log(2)), percentage_cnt_neg_list,
            color='black')
# axs[1].plot((np.log(eta_list) / np.log(2)), pi_list, color='tab:red',
#             linestyle='--')

axs[1].invert_xaxis()
plt.savefig('{}_{}_{}.pdf'.format(pg_method, optim_type, stepsize_type))
exit()
