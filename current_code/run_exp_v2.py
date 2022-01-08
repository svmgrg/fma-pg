import pdb
import time
import argparse
import os
import json

from environments import *
from utils_v2 import *

# $ python run_exp.py --num_outer_loops 2000 --num_inner_loops 1 --pg_alg 'TRPO_KL_LS' --eta -1 --alpha -1 --epsilon -1 --delta -1 --decay_factor 0.9 --use_analytical_grad 0 --zeta 128 --armijo_constant 0 --max_backtracking_iters -1 --flag_warm_start 1 --warm_start_beta_init 10 --warm_start_beta_factor 10 

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-pa', '--pg_method', required=True, type=str)
parser.add_argument('-nol', '--num_outer_loop', required=True, type=int)
parser.add_argument('-nil', '--num_inner_loop', required=True, type=int)
parser.add_argument('-fag', '--FLAG_ANALYTICAL_GRAD', required=True, type=int)
parser.add_argument('-fsis', '--FLAG_SAVE_INNER_STEPS', required=True, type=int)
parser.add_argument('-am', '--alpha_max', required=True, type=float)
parser.add_argument('-fws', '--FLAG_WARM_START', required=True, type=int)
parser.add_argument('-wsf', '--warm_start_factor', required=True, type=float)
parser.add_argument('-mbs', '--max_backtracking_steps', required=True, type=int)
parser.add_argument('-ot', '--optim_type', required=True, type=str)
parser.add_argument('-st', '--stepsize_type', required=True, type=str)
parser.add_argument('-eta', '--eta', required=True, type=float)
parser.add_argument('-eps', '--epsilon', required=True, type=float)
parser.add_argument('-del', '--delta', required=True, type=float)
parser.add_argument('-af', '--alpha_fixed', required=True, type=float)
parser.add_argument('-df', '--decay_factor', required=True, type=float)
parser.add_argument('-ac', '--armijo_constant', required=True, type=float)

args = parser.parse_args()

pg_method = args.pg_method
num_outer_loop = args.num_outer_loop
num_inner_loop = args.num_inner_loop if args.num_inner_loop >= 0 else None
FLAG_ANALYTICAL_GRAD = False if args.FLAG_ANALYTICAL_GRAD == 0 else True
FLAG_SAVE_INNER_STEPS = False if args.FLAG_SAVE_INNER_STEPS == 0 else True
alpha_max = args.alpha_max
FLAG_WARM_START = False if args.FLAG_WARM_START == 0 else True
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
armijo_constant = args.armijo_constant if args.armijo_constant >= 0 else None

# folder details
folder_name = 'fmaPG_exp/CliffWorld_{}'.format(pg_method)
os.makedirs(folder_name, exist_ok='True')

#----------------------------------------------------------------------
# create the environment
#----------------------------------------------------------------------
gamma = 0.9
episode_cutoff_length = 100
reward_noise = 0

env = CliffWorld(P=P, r=r, mu=mu, terminal_states=terminal_states,
                 gamma=gamma, episode_cutoff_length=episode_cutoff_length,
                 reward_noise=reward_noise)

#----------------------------------------------------------------------
# learning curves against the number of outer loop iterations
#----------------------------------------------------------------------
tic = time.time()
vpi_list_outer, cnt_neg_list, vpi_list_inner, pi = run_experiment(
    env=env, pg_method=pg_method, num_outer_loop=num_outer_loop,
    num_inner_loop=num_inner_loop, FLAG_ANALYTICAL_GRAD=FLAG_ANALYTICAL_GRAD,
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
dat = dict()
dat['vpi_outer_list'] = vpi_list_outer
dat['cnt_neg_list'] = cnt_neg_list
dat['vpi_inner_list'] = vpi_list_inner
dat['pi'] = pi

filename='{}/numOuterLoop_{}__numInnerLoop_{}__flagAnalyticalGrad_{}'\
    '__flagSaveInnerSteps_{}__alphaMax_{}__flagWarmStart_{}__warmStartFactor_{}'\
    '__maxBacktrackingSteps_{}__optimType_{}__eta_{}__epsilon_{}__delta_{}'\
    '__alphaFixed_{}__decayFactor_{}__armijoConstant_{}'.format(
        num_outer_loop, num_inner_loop, FLAG_ANALYTICAL_GRAD,
        FLAG_SAVE_INNER_STEPS, alpha_max, FLAG_WARM_START, warm_start_factor,
        max_backtracking_steps, optim_type, stepsize_type, eta, epsilon, delta,
        alpha_fixed, decay_factor, armijo_constant)

with open(filename, 'w') as fp:
    json.dump(dat, fp)

#----------------------------------------------------------------------
# misc plotting
#----------------------------------------------------------------------
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# plot_grid(axs[0])
# plot_policy(axs[0], pi)
# axs[1].plot(vpi_list_outer)
# v_star = np.dot(env.calc_v_star(), env.mu)
# axs[1].set_ylim([-1, v_star + 0.05])
# plt.savefig('lc_{}_{}_env.pdf'.format(pg_method, eta))
# plt.show()  
# plt.close()


