import pdb
import time
import argparse
import os
import json

from environments import *
from utils import *

# $ python run_exp.py --num_outer_loops 2000 --num_inner_loops 1 --pg_alg 'TRPO_KL_LS' --eta -1 --alpha -1 --epsilon -1 --delta -1 --decay_factor 0.9 --use_analytical_grad 0 --zeta 128 --armijo_constant 0 --max_backtracking_iters -1 --flag_warm_start 1 --warm_start_beta_init 10 --warm_start_beta_factor 10 

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--num_outer_loops', required=True, type=int)
parser.add_argument('-m', '--num_inner_loops', required=True, type=int)
parser.add_argument('-alg', '--pg_alg', required=True, type=str)
parser.add_argument('-e', '--eta', required=True, type=float)
parser.add_argument('-a', '--alpha', required=True, type=float)
parser.add_argument('-eps', '--epsilon', required=True, type=float)
parser.add_argument('-d', '--delta', required=True, type=float)
parser.add_argument('-dec', '--decay_factor', required=True, type=float)
parser.add_argument('-tg', '--use_analytical_grad', required=True, type=int)
parser.add_argument('-z', '--zeta', required=True, type=float)
parser.add_argument('-arm', '--armijo_constant', required=True, type=float)
parser.add_argument('-mbi', '--max_backtracking_iters', required=True, type=int)
parser.add_argument('-ws', '--flag_warm_start', required=True, type=int)
parser.add_argument('-wsbi', '--warm_start_beta_init', required=True,
                    type=float)
parser.add_argument('-wsbf', '--warm_start_beta_factor', required=True,
                    type=float)

args = parser.parse_args()

pg_method = args.pg_alg
num_outer_loops = args.num_outer_loops
num_inner_loops = args.num_inner_loops if args.num_inner_loops >= 0 else None
eta = args.eta if args.eta >= 0 else None
alpha = args.alpha if args.alpha >= 0 else None
epsilon = args.epsilon if args.epsilon >= 0 else None
delta = args.delta if args.delta >= 0 else None
decay_factor = args.decay_factor if args.decay_factor >= 0 else None
FLAG_ANALYTICAL_GRADIENT = False if args.use_analytical_grad == 0 else True
zeta = args.zeta if args.zeta >= 0 else None
armijo_constant = args.armijo_constant if args.armijo_constant >= 0 else None
max_backtracking_iters = args.max_backtracking_iters \
    if args.max_backtracking_iters >= 0 else None
FLAG_BETA_WARM_START = False if args.flag_warm_start == 0 else True
warm_start_beta_init = args.warm_start_beta_init \
    if args.warm_start_beta_init >= 0 else None
warm_start_beta_factor = args.warm_start_beta_factor \
    if args.warm_start_beta_factor >= 0 else None

FLAG_TRUE_ADVANTAGE = True
num_traj_estimate_adv = None 
adv_estimate_alg = None
adv_estimate_stepsize = None
FLAG_SAVE_INNER_STEPS = False

# folder details
folder_name = 'fmaPG_exp/CliffWorld_{}'.format(pg_method)
os.makedirs(folder_name, exist_ok='True')

#----------------------------------------------------------------------
# create the environment
#----------------------------------------------------------------------
gamma = 0.9
episode_cutoff_length = 1000
reward_noise = 0

env = CliffWorld(P=P, r=r, mu=mu, terminal_states=terminal_states,
                 gamma=gamma, episode_cutoff_length=episode_cutoff_length,
                 reward_noise=reward_noise)

#----------------------------------------------------------------------
# learning curves against the number of outer loop iterations
#----------------------------------------------------------------------
tic = time.time()
vpi_list_outer, vpi_list_inner, pi = run_experiment(
    env=env, pg_method=pg_method, num_iters=num_outer_loops,
    eta=eta, delta=delta, decay_factor=decay_factor, epsilon=epsilon,
    FLAG_ANALYTICAL_GRADIENT=FLAG_ANALYTICAL_GRADIENT,
    num_inner_updates=num_inner_loops, alpha=alpha,
    FLAG_TRUE_ADVANTAGE=FLAG_TRUE_ADVANTAGE,
    adv_estimate_alg=adv_estimate_alg,
    num_traj_estimate_adv=num_traj_estimate_adv,
    adv_estimate_stepsize=adv_estimate_stepsize,
    FLAG_SAVE_INNER_STEPS=FLAG_SAVE_INNER_STEPS, zeta=zeta,
    armijo_constant=armijo_constant,
    max_backtracking_iters=max_backtracking_iters,
    FLAG_BETA_WARM_START=FLAG_BETA_WARM_START,
    warm_start_beta_init=warm_start_beta_init,
    warm_start_beta_factor=warm_start_beta_factor)
print('Total time taken: {}'.format(time.time() - tic))

dat = dict()
dat['vpi_outer_list'] = vpi_list_outer.tolist()
if vpi_list_inner is not None:
    vpi_list_inner = vpi_list_inner.tolist()
dat['vpi_inner_list'] = vpi_list_inner
dat['pi'] = pi.tolist()

filename='{}/numOuterLoops_{}__numInnerLoops_{}__eta_{}__alpha_{}'\
    '__epsilon_{}__delta_{}__decayFactor_{}__analyticalGrad_{}__zeta_{}'\
    '__armijoConstant_{}__maxBacktrackingIters_{}__FlagBetaWarmStart_{}'\
    '__wsbI_{}__wsbF_{}'.format(
        folder_name, num_outer_loops, num_inner_loops, eta, alpha, epsilon,
        delta, decay_factor, FLAG_ANALYTICAL_GRADIENT, zeta, armijo_constant,
        max_backtracking_iters, FLAG_BETA_WARM_START,
        warm_start_beta_init, warm_start_beta_factor)
with open(filename, 'w') as fp:
    json.dump(dat, fp)
    
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# plot_grid(axs[0])
# plot_policy(axs[0], pi)
# axs[1].plot(vpi_list_outer)
# v_star = np.dot(env.calc_v_star(), env.mu)
# axs[1].set_ylim([-1, v_star + 0.05])
# plt.savefig('lc_{}_{}_env.pdf'.format(pg_method, eta))
# plt.show()  
# plt.close()


