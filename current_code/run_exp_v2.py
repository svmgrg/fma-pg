import pdb
import time
import argparse
import os
import json

from environments import *
from utils_v2 import *

# $ python run_exp_v2.py --pg_method 'sPPO' --num_outer_loop 2000 --num_inner_loop -1 --FLAG_SAVE_INNER_STEPS 'False' --alpha_max -1 --FLAG_WARM_START -1 --warm_start_factor -1 --max_backtracking_steps -1 --optim_type 'analytical' --stepsize_type 'fixed' --eta 1 --epsilon -1 --delta -1 --alpha_fixed -1 --decay_factor -1 --armijo_const -1

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment', required=True, type=str)
parser.add_argument('-pa', '--pg_method', required=True, type=str)
parser.add_argument('-nol', '--num_outer_loop', required=True, type=int)
parser.add_argument('-nil', '--num_inner_loop', required=True, type=int)
parser.add_argument('-fsis', '--FLAG_SAVE_INNER_STEPS', required=True, type=str)
parser.add_argument('-am', '--alpha_max', required=True, type=float)
parser.add_argument('-fws', '--FLAG_WARM_START', required=True, type=str)
parser.add_argument('-wsf', '--warm_start_factor', required=True, type=float)
parser.add_argument('-mbs', '--max_backtracking_steps', required=True, type=int)
parser.add_argument('-ot', '--optim_type', required=True, type=str)
parser.add_argument('-st', '--stepsize_type', required=True, type=str)
parser.add_argument('-eta', '--eta', required=True, type=float)
parser.add_argument('-eps', '--epsilon', required=True, type=float)
parser.add_argument('-del', '--delta', required=True, type=float)
parser.add_argument('-af', '--alpha_fixed', required=True, type=float)
parser.add_argument('-df', '--decay_factor', required=True, type=float)
parser.add_argument('-ac', '--armijo_const', required=True, type=float)

args = parser.parse_args()

env_name = args.environment
pg_method = args.pg_method
num_outer_loop = args.num_outer_loop
num_inner_loop = args.num_inner_loop if args.num_inner_loop >= 0 else None
FLAG_SAVE_INNER_STEPS = True if args.FLAG_SAVE_INNER_STEPS == 'True' else False
alpha_max = args.alpha_max if args.alpha_max >=0 else None
FLAG_WARM_START = True if args.FLAG_WARM_START == 'True' else False
warm_start_factor = args.warm_start_factor \
    if args.warm_start_factor >= 0 else None
max_backtracking_steps = args.max_backtracking_steps \
    if args.max_backtracking_steps >=0 else None
optim_type = args.optim_type
stepsize_type = args.stepsize_type
eta = args.eta if args.eta >= 0 else None
epsilon = args.epsilon if args.epsilon >= 0 else None
delta = args.delta if args.delta >= 0 else None
alpha_fixed = args.alpha_fixed if args.alpha_fixed >= 0 else None
decay_factor = args.decay_factor if args.decay_factor >= 0 else None
armijo_const = args.armijo_const if args.armijo_const >= 0 else None

#----------------------------------------------------------------------
# create the environment
#----------------------------------------------------------------------
gamma = 0.9
episode_cutoff_length = 100
reward_noise = 0

if env_name == 'CliffWorld':
    P = P_CliffWorld
    r = r_CliffWorld
    mu = mu_CliffWorld
    terminal_states = terminal_states_CliffWorld
elif env_name == 'DeepSeaTreasure':
    P = P_DeepSeaTreasure
    r = r_DeepSeaTreasure
    mu = mu_DeepSeaTreasure
    terminal_states = terminal_states_DeepSeaTreasure
else:
    raise NotImplementedError()

env = TabularMDP(P=P, r=r, mu=mu, terminal_states=terminal_states,
                 gamma=gamma, episode_cutoff_length=episode_cutoff_length,
                 reward_noise=reward_noise)

#----------------------------------------------------------------------
# learning curves against the number of outer loop iterations
#----------------------------------------------------------------------
tic = time.time()
dat = run_experiment(
    env=env, pg_method=pg_method, num_outer_loop=num_outer_loop,
    num_inner_loop=num_inner_loop,
    FLAG_SAVE_INNER_STEPS=FLAG_SAVE_INNER_STEPS, alpha_max=alpha_max,
    FLAG_WARM_START=FLAG_WARM_START, warm_start_factor=warm_start_factor,
    max_backtracking_steps=max_backtracking_steps, optim_type=optim_type,
    stepsize_type=stepsize_type, eta=eta, epsilon=epsilon, delta=delta,
    alpha_fixed=alpha_fixed, decay_factor=decay_factor,
    armijo_const=armijo_const)
print('Total time taken: {}'.format(time.time() - tic))

#----------------------------------------------------------------------
# save the data
#----------------------------------------------------------------------
folder_name = 'fmapg_DAT/{}_{}_{}_{}'.format(
    env_name, pg_method, optim_type, stepsize_type)
os.makedirs(folder_name, exist_ok='True')

filename='{}/nmOtrLp_{}__nmInrLp_{}'\
    '__SvInrStps_{}__alphMx_{}__WrmStr_{}__wrmStrFac_{}'\
    '__mxBktStps_{}__eta_{}__eps_{}'\
    '__del_{}__alphFxd_{}__dcyFac_{}__armjoCnst_{}'.format(
        folder_name, num_outer_loop, num_inner_loop,
        FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START, warm_start_factor,
        max_backtracking_steps, eta, epsilon,
        delta, alpha_fixed, decay_factor, armijo_const)

with open(filename, 'w') as fp:
    json.dump(dat, fp)

#----------------------------------------------------------------------
# misc plotting
#----------------------------------------------------------------------
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 3)

# axs[0].plot(dat['vpi_outer_list'])
# axs[1].plot(np.log10(np.array(dat['grad_lpi_inner_list'])))
# axs[2].plot(dat['grad_jpi_outer_list'])

# if dat['vpi_outer_list'][-1] > 0:
#     axs[0].set_ylim([0, 0.9**6 + 0.05])
# plt.show()

# pdb.set_trace()
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# plot_grid(axs[0])
# plot_policy(axs[0], pi)
# axs[1].plot(vpi_list_outer)
# v_star = np.dot(env.calc_v_star(), env.mu)
# axs[1].set_ylim([-1, v_star + 0.05])
# plt.savefig('lc_{}_{}_env.pdf'.format(pg_method, eta))
# plt.show()  
# plt.close()

# python run_exp_v2.py --pg_method 'TRPO' --num_outer_loop 2000 --num_inner_loop 10 --FLAG_SAVE_INNER_STEPS 'True' --alpha_max 100 --FLAG_WARM_START 'True' --warm_start_factor 2 --max_backtracking_steps 100 --optim_type 'regularized' --stepsize_type 'line_search' --eta 0.1 --epsilon -1 --delta -1 --alpha_fixed -1 --decay_factor 0.9 --armijo_const 0.5

# python run_exp_v2.py --pg_method 'sPPO' --num_outer_loop 2000 --num_inner_loop 1000 --FLAG_SAVE_INNER_STEPS 'False' --alpha_max -1 --FLAG_WARM_START 'False' --warm_start_factor -1 --max_backtracking_steps -1 --optim_type 'regularized' --stepsize_type 'fixed' --eta 0.03125 --epsilon -1 --delta -1 --alpha_fixed 0.125 --decay_factor -1 --armijo_const -1


# constrained

# python run_exp_v2.py --pg_method 'MDPO' --num_outer_loop 2000 --num_inner_loop 10 --FLAG_SAVE_INNER_STEPS 'False' --alpha_max -1 --FLAG_WARM_START 'False' --warm_start_factor -1 --max_backtracking_steps 1000 --optim_type 'constrained' --stepsize_type 'line_search' --eta -1 --epsilon -1 --delta 0.1 --alpha_fixed -1 --decay_factor 0.9 --armijo_const 0
