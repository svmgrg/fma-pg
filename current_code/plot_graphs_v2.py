import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import argparse
import copy

VSTAR = 0.9**6
color_dict = {'sPPO': 'blue', 'MDPO': 'red', 'PPO': 'green',
              'TRPO': 'tab:purple', 'TRPO_KL': 'tab:purple',
              'TRPO_KL_LS': 'cyan'}
linewidth = 1

num_inner_loop_list = [1, 10, 100, 1000]

eta_list = [2**i for i in range(-13, 4, 1)]
delta_list = [2**i for i in range(-13, 14, 2)]

alpha_list = [2**i for i in range(-13, 4, 1)]
armijo_const_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

pg_method = 'TRPO'
folder_name = 'fmapg_DAT/CliffWorld_{}'.format(pg_method)
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

def plot_sensitivity(ax, num_inner_loop, pg_method, final_perf_idx=-1, 
                     linewidth=1):
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
                '__mxBktStps_{}__optmTyp_{}__stpTyp_{}__eta_{}__eps_{}'\
                '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                    folder_name, num_outer_loop, num_inner_loop,
                    FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                    warm_start_factor, max_backtracking_steps, optim_type,
                    stepsize_type, eta, epsilon, delta, alpha_fixed,
                    decay_factor, armijo_const)
            with open(filename, 'r') as fp:
                dat = json.load(fp)
            final_perf_list.append(dat['vpi_outer_list'][final_perf_idx])

        ax.plot((np.log(eta_list) / np.log(2)), final_perf_list,
                color=plt_color, label=r'$\alpha:{}$'.format(alpha_fixed),
                linewidth=linewidth)
        ax.plot(np.log(eta_list) / np.log(2), [VSTAR] * len(eta_list),
            color='black', linestyle=':', linewidth=0.5)
        ax.invert_xaxis()
    
num_rows = len(num_inner_loop_list)
fig, axs = plt.subplots(1, num_rows, figsize=(3 * num_rows, 1.7))
final_perf_idx = -1
for row, num_inner_loop in enumerate(num_inner_loop_list):
    plot_sensitivity(ax=axs[row], num_inner_loop=num_inner_loop,
                     pg_method=pg_method, final_perf_idx=final_perf_idx,
                     linewidth=linewidth)
    axs[row].spines['right'].set_visible(False)
    axs[row].spines['top'].set_visible(False)
    axs[row].set_ylim([0, 0.6])
    # axs[row].set_title('m={}'.format(num_inner_loop))

plt.savefig('{}_{}.pdf'.format(pg_method, optim_type))
exit()






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

