import os
import pdb

num_inner_loop_list = [1, 10, 100]

# inner lists
eta_list = [2**i for i in range(-13, -2, 1)]
delta_list =  [2**i for i in range(-13, +0, 2)] # [2**i for i in range(-24, 0, 2)] # 
epsilon_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

# outer lists
alpha_list = [2**i for i in range(-13, 4, 1)] # [2**i for i in range(-13, 4, 1)]
alpha_list_big = [2**i for i in range(-13, 4, 1)] # + [0.199, 0.2, 0.201]

armijo_const_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

env_name = 'CliffWorld' # 'DeepSeaTreasure' # 
pg_method = 'sPPO' #

num_outer_loop = 2000
FLAG_SAVE_INNER_STEPS = False
alpha_max = 10.0 # 10.0 # None # 100000.0 # 10.0 # 
FLAG_WARM_START = True # False #
warm_start_factor = 2.0 # None # 
max_backtracking_steps = 1000 # None # 
optim_type = 'regularized' # 'constrained' # 
stepsize_type = 'line_search' # 'line_search' # 
alpha_fixed = None
decay_factor = 0.9 # None #

folder_name = 'fmapg_DAT2/{}_{}_{}_{}'.format(
    env_name, pg_method, optim_type, stepsize_type)

if optim_type == 'analytical':
    outer_list = [None]
elif stepsize_type == 'fixed':
    outer_list = alpha_list
elif stepsize_type == 'line_search':
    outer_list = armijo_const_list

if pg_method == 'PPO':
    inner_list = epsilon_list
elif optim_type in ['regularized', 'analytical']:
    inner_list = eta_list
elif optim_type == 'constrained':
    inner_list = delta_list
eta = delta = epsilon = None

counts_missing = 0

for num_inner_loop in num_inner_loop_list:
    for outer_idx in outer_list:
        alpha_fixed = outer_idx if stepsize_type == 'fixed' else None
        armijo_const = float(outer_idx) \
            if stepsize_type == 'line_search' else None

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

            if not os.path.isfile(filename):
                counts_missing += 1
                print(eta, armijo_const)
            
print(counts_missing)
