import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

color_dict = {'sPPO': 'blue', 'MDPO': 'red', 'PPO': 'green',
              'TRPO': 'orange', 'TRPO_KL': 'tab:purple',
              'TRPO_KL_LS': 'cyan'}
linewidth = 1

num_inner_loop_list = [10, 100, 1000] #, 10000]

# inner lists
eta_list = [2**i for i in range(-13, -4, 2)]
delta_list =  [2**i for i in range(-13, +0, 2)] # [2**i for i in range(-24, 0, 2)] # 
epsilon_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

# outer lists
alpha_list = [2**i for i in range(-13, 4, 1)] # [2**i for i in range(-13, 4, 1)]
alpha_list_big = [2**i for i in range(-13, 4, 1)] # + [0.199, 0.2, 0.201]

armijo_const_list = [0.0] # 0, 0.1, 0.5] #, 0.001, 0.01, 0.1, 0.5]

env_name = 'CliffWorld' # 'DeepSeaTreasure' # 
pg_method = 'sPPO' #
PLOT_TYPE = 'vpi_outer' # 'grad_jpi_outer' #

if env_name == 'DeepSeaTreasure':
    VSTAR = 0.9**3
elif env_name == 'CliffWorld':
    VSTAR = 0.9**6

# num_outer_loop = 2000
# FLAG_SAVE_INNER_STEPS = False
# alpha_max = None # 100000.0 # 10.0 # 
# FLAG_WARM_START = False
# warm_start_factor = None # 2.0 # None
# max_backtracking_steps = None # 1000 # 
# optim_type = 'regularized' # 'constrained' # 
# stepsize_type = 'fixed' # 'line_search' # 
# alpha_fixed = None
# decay_factor = None # 0.9 #

num_outer_loop = 2000
FLAG_SAVE_INNER_STEPS = False
alpha_max = 10.0 # None # 100000.0 # 10.0 # 
FLAG_WARM_START = False
warm_start_factor = None # 2.0 # None
max_backtracking_steps = 1000 # None # 
optim_type = 'regularized' # 'constrained' # 
stepsize_type = 'line_search' # 'line_search' # 
alpha_fixed = None
decay_factor = 0.9 # None # 

# num_outer_loop = 2000
# FLAG_SAVE_INNER_STEPS = False
# alpha_max = 100000.0 # None # 10.0 # 
# FLAG_WARM_START = False # True # 
# warm_start_factor = None # 2.0 # 
# max_backtracking_steps = 1000 # None # 
# optim_type = 'constrained' # 'regularized' # 
# stepsize_type = 'line_search' # 'fixed' # 
# alpha_fixed = None
# decay_factor = 0.9 # None #

folder_name = 'fmapg_DAT/{}_{}_{}_{}'.format(
    env_name, pg_method, optim_type, stepsize_type)

num_cols = len(num_inner_loop_list)

def plot_sensitivity(ax, num_inner_loop, pg_method, final_perf_idx=-1, 
                     plot_type=None, linewidth=1):
    assert plot_type in ['vpi_outer', 'grad_lpi_inner', 'grad_jpi_outer']
    
    c1 = np.array(mpl.colors.to_rgb(color_dict[pg_method]))
    c2 = np.array(mpl.colors.to_rgb('white'))

    if optim_type == 'analytical':
        outer_list = [None]
    elif stepsize_type == 'fixed':
        outer_list = alpha_list
    elif stepsize_type == 'line_search':
       outer_list = armijo_const_list

    for outer_idx, clr_mix in zip(outer_list, range(len(outer_list))):
        alpha_fixed = outer_idx if stepsize_type == 'fixed' else None
        if alpha_fixed is not None:
            alpha_fixed = float(alpha_fixed)
        
        armijo_const = float(outer_idx) \
            if stepsize_type == 'line_search' else None

        mix = (clr_mix + 1) / (len(outer_list))
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

        final_perf_list = []

        if pg_method == 'PPO':
            inner_list = epsilon_list
        elif optim_type in ['regularized', 'analytical']:
            inner_list = eta_list
        elif optim_type == 'constrained':
            inner_list = delta_list

        eta = delta = epsilon = None
        for inner_idx in inner_list:
            if pg_method == 'PPO':
                epsilon = float(inner_idx)
            elif optim_type in ['regularized', 'analytical']:
                eta = float(inner_idx)
            elif optim_type == 'constrained':
                delta = float(inner_idx)

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

        if pg_method == 'PPO':
            x_axis = inner_list
        else:
            x_axis = np.log(inner_list) / np.log(2)
        ax.plot(x_axis, final_perf_list,
                color=plt_color, label=r'$\alpha:{}$'.format(alpha_fixed),
                linewidth=linewidth)
        
        if plot_type == 'vpi_outer':
            ax.plot(x_axis, [VSTAR] * len(inner_list),
                    color='black', linestyle=':', linewidth=0.5)
            if env_name == 'CliffWorld':
                ax.set_ylim([0, 0.6])
            elif env_name == 'DeepSeaTreasure':
                ax.set_ylim([0, 0.75])
        elif plot_type == 'grad_jpi_outer':
            ax.set_yscale('log')

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                        box.width, box.height * 0.9])
    # if pg_method != 'TRPO':
    #     ax.invert_xaxis()



def plot_sensitivity_best_alpha(ax, num_inner_loop, pg_method,
                                final_perf_idx=-1,  plot_type=None,
                                linewidth=1):
    assert plot_type == 'vpi_outer'
       
    if pg_method == 'PPO':
        outer_list = epsilon_list
    elif optim_type in ['regularized', 'analytical']:
        outer_list = eta_list
    elif optim_type == 'constrained':
        outer_list = delta_list

    final_perf_list = []
    eta = delta = epsilon = None    
    for outer_idx in outer_list:
        if pg_method == 'PPO':
            epsilon = float(outer_idx)
        elif optim_type in ['regularized', 'analytical']:
            eta = float(outer_idx)
        elif optim_type == 'constrained':
            delta = float(outer_idx)

        if optim_type == 'analytical':
            inner_list = [None]
        elif stepsize_type == 'fixed':
            # get rid of the weird kink in the sensitivity plot for sPPO
            if pg_method == 'sPPO' and eta == 0.015625 and num_inner_loop == 10:
                inner_list = alpha_list_big
            else:
                inner_list = alpha_list
        elif stepsize_type == 'line_search':
            inner_list = armijo_const_list

        max_perf = -np.inf
        best_alpha = None
            
        for inner_idx in inner_list:
            alpha_fixed = inner_idx if stepsize_type == 'fixed' else None
            if alpha_fixed is not None:
                alpha_fixed = float(alpha_fixed)
                
            armijo_const = float(inner_idx) \
                if stepsize_type == 'line_search' else None

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

            final_perf = dat[plot_type + '_list'][final_perf_idx]
            if  final_perf > max_perf:
                best_alpha = inner_idx
                max_perf = final_perf
                
        final_perf_list.append(max_perf)
        
    x_axis = outer_list if pg_method == 'PPO' \
        else np.log(outer_list) / np.log(2)
    ax.plot(x_axis, final_perf_list, color=color_dict[pg_method],
            linewidth=linewidth)
    ax.plot(x_axis, [VSTAR] * len(outer_list),
            color='black', linestyle=':', linewidth=0.5)
    if env_name == 'CliffWorld':
        ax.set_ylim([0, 0.6])
    elif env_name == 'DeepSeaTreasure':
        ax.set_ylim([0, 0.75])
        
    if optim_type != 'constrained':
        ax.invert_xaxis()
    # if pg_method == 'TRPO':
    #     ax.invert_xaxis()

#======================================================================
# Plot sensitivity plots
#======================================================================
if optim_type == 'analytical':
    num_inner_loop_list = [None]
num_cols = len(num_inner_loop_list)
fig, axs = plt.subplots(1, num_cols, sharey=True, figsize=(3 * num_cols, 1.7))
final_perf_idx = -1
for col, num_inner_loop in enumerate(num_inner_loop_list):
    ax = axs[col] if num_cols > 1 else axs
    
    plot_sensitivity_best_alpha(
        ax=ax, num_inner_loop=num_inner_loop, pg_method=pg_method,
        final_perf_idx=final_perf_idx, plot_type=PLOT_TYPE,
        linewidth=linewidth)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('m={}'.format(num_inner_loop))

    # if col == 2:
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
    #                     fancybox=True, shadow=True, ncol=6)
ax = axs[0] if num_cols > 1 else axs
# plt.show()
plt.savefig('final_trpo_party/best_alpha/{}_{}_{}_{}_{}_sensitivity.pdf'.format(
    env_name, pg_method, optim_type, stepsize_type, PLOT_TYPE))
exit()

#======================================================================
# Plot best learning curves
#======================================================================
def find_best_param_config(folder_name, pg_method, num_inner_loop, 
                           optim_type, stepsize_type, decay_factor, alpha_max,
                           max_backtracking_steps, best_config_type,
                           final_perf_idx=-1):
    assert best_config_type in ['min_grad_jpi', 'max_jpi']
    
    min_grad_jpi = np.inf
    max_vpi = -np.inf
    best_alpha_fixed = best_armijo_const = best_eta = best_delta = best_epsilon\
        = None
        
    outer_list = alpha_list if stepsize_type == 'fixed' else armijo_const_list
    for outer_idx, clr_mix in zip(outer_list, range(len(outer_list))):
        alpha_fixed = float(outer_idx) if stepsize_type == 'fixed' else None
        armijo_const = float(outer_idx) \
            if stepsize_type == 'line_search' else None

        if pg_method == 'PPO':
            inner_list = epsilon_list
        elif optim_type == 'regularized':
            # inner_list = eta_list
            if pg_method == 'sPPO':
                inner_list = [(1 - 0.9) / 101]
            elif pg_method == 'MDPO':
                inner_list = [(1 - 0.9)**3 / (2 * 0.9 * 4) * 101]
        elif optim_type == 'constrained':
            inner_list = delta_list
            
        eta = delta = epsilon = None
        for inner_idx in inner_list:
            if pg_method == 'PPO':
                epsilon = float(inner_idx)
            elif optim_type == 'regularized':
                eta = float(inner_idx)
            elif optim_type == 'constrained':
                delta = float(inner_idx)

            filename='{}/nmOtrLp_{}__nmInrLp_{}'\
                '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
                '__mxBktStps_{}__eta_{}__eps_{}'\
                '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
                    folder_name, num_outer_loop, num_inner_loop,
                    FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START,
                    warm_start_factor, max_backtracking_steps, eta, epsilon,
                    delta, alpha_fixed, dcy_fac, armijo_const)
            with open(filename, 'r') as fp:
                dat = json.load(fp)

            grad = dat['grad_jpi_outer_list'][final_perf_idx]
            perf = dat['vpi_outer_list'][final_perf_idx]

            if best_config_type == 'min_grad_jpi':
                best_param_condition = grad < min_grad_jpi
            elif best_config_type == 'max_jpi':
                best_param_condition = perf > max_vpi

            if best_param_condition: # grad < min_grad_jpi:
                min_grad_jpi = grad
                max_vpi = perf
                best_alpha_fixed = alpha_fixed
                best_armijo_const = armijo_const
                best_eta = eta
                best_delta = delta
                best_epsilon = epsilon

    return best_alpha_fixed, best_armijo_const, best_eta,\
        best_delta, best_epsilon

num_inner_loop = 1000
best_config_type = 'max_jpi' # 'min_grad_jpi' # 
fig, axs = plt.subplots(1, 2, figsize=(3 * 2, 1.7))

for pg_method in ['sPPO', 'MDPO', 'PPO']:
    if pg_method == 'TRPO':
        optim_type = 'constrained'
        stepsize_type = 'line_search'
        dcy_fac = decay_factor
        alph_mx = alpha_max
        mx_bkt_stps = max_backtracking_steps
    else:
        optim_type = 'regularized'
        stepsize_type = 'fixed'
        dcy_fac = None
        alph_mx = None
        mx_bkt_stps = None

    folder_name = 'fmapg_DAT/CliffWorld_{}_{}_{}'.format(
        pg_method, optim_type, stepsize_type)
        
    best_alpha_fixed, best_armijo_const, best_eta, best_delta, best_epsilon = \
        find_best_param_config(
            folder_name=folder_name, pg_method=pg_method,
            num_inner_loop=num_inner_loop, optim_type=optim_type,
            stepsize_type=stepsize_type, decay_factor=dcy_fac,
            alpha_max=alph_mx, max_backtracking_steps=mx_bkt_stps,
            best_config_type=best_config_type, final_perf_idx=-1)
    
    print(pg_method, best_alpha_fixed, 'eta', best_eta, 'delta', best_delta,
          'epsilon', best_epsilon)

    filename='{}/nmOtrLp_{}__nmInrLp_{}'\
        '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
        '__mxBktStps_{}__eta_{}__eps_{}'\
        '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
            folder_name, num_outer_loop, num_inner_loop,
            FLAG_SAVE_INNER_STEPS, alph_mx, FLAG_WARM_START,
            warm_start_factor, mx_bkt_stps, best_eta, best_epsilon,
            best_delta, best_alpha_fixed, dcy_fac, best_armijo_const)
    with open(filename, 'r') as fp:
        dat = json.load(fp)

    axs[0].plot(dat['vpi_outer_list'], color=color_dict[pg_method])
    axs[1].plot(dat['grad_jpi_outer_list'], color=color_dict[pg_method])

    axs[0].set_ylim([-0.1, 0.6])
    axs[1].set_yscale('log')

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

plt.savefig('{}_m{}_learning_curve.pdf'.format(
    best_config_type, num_inner_loop))
exit()

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
