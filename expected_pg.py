import numpy as np
import matplotlib.pyplot as plt
import pdb

from src.envs.gridworld_mdp import cliff_gw

#----------------------------------------------------------------------
# plotting functions
#----------------------------------------------------------------------
def plot_grid(ax, xlim=7, ylim=7):
    x_center = np.arange(xlim) + 0.5
    y_center = np.arange(ylim) + 0.5

    for x in x_center:
        for y in y_center:
            if int((x - 0.5) * ylim + (y - 0.5)) == 8:
                ax.scatter(x, y, s=200, color='tab:blue', marker='s',
                           label='start', alpha=0.5)
            elif int((x - 0.5) * ylim + (y - 0.5)) in [9, 10, 11]:
                ax.scatter(x, y, s=200, color='tab:red', marker='x',
                           label='chasm', alpha=0.5)
            elif int((x - 0.5) * ylim + (y - 0.5)) == 12:
                ax.scatter(x, y, s=200, color='tab:green', marker='o',
                           label='end', alpha=0.5)
            else:
                ax.scatter(x, y, s=0.1, color='purple')

    for x in range(xlim + 1):
        ax.plot((x, x), (0, ylim), linewidth=0.5, color='black')

    for y in range(ylim + 1):
        ax.plot((0, xlim), (y, y), linewidth=0.5, color='black')

       
def plot_policy(ax, pi, xlim=7, ylim=7):
    for s_x in range(xlim):
        for s_y in range(ylim):
            x_center = s_x + 0.5
            y_center = s_y + 0.5
            s = s_x * ylim + s_y
            diff = pi[s] / 2

            # down
            ax.quiver(x_center, y_center, 0, -diff[0],
                      color='tab:orange', width=0.007, headwidth=2,
                      headlength=4,
                      scale=1, scale_units='xy', linewidth=0.1)
            # up
            ax.quiver(x_center, y_center, 0, +diff[1],
                      color='tab:blue', width=0.007, headwidth=2, headlength=4,
                      scale=1, scale_units='xy', linewidth=0.1)
            # left
            ax.quiver(x_center, y_center, -diff[2], 0,
                      color='tab:red', width=0.007, headwidth=2, headlength=4,
                      scale=1, scale_units='xy', linewidth=0.1)
            # right
            ax.quiver(x_center, y_center, +diff[3], 0,
                      color='tab:green', width=0.007, headwidth=2, headlength=4,
                      scale=1, scale_units='xy', linewidth=0.1)



#----------------------------------------------------------------------
# utility functions
#----------------------------------------------------------------------

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum(1).reshape(-1, 1)
    return out

# generate a uniformly random tabular policy (direct representation)
def generate_uniform_policy(num_states, num_actions):
    return np.ones((num_states, num_actions)) / num_actions

# generate action preferences (for softmax policy representation)
def init_theta(num_states, num_actions, theta_init=0):
    return theta_init * np.ones((num_states, num_actions))

# updates the policy using the closed form analytical FMA-PG update
# refer to the "notes_fmapg.pdf" document; or the references given below
def analytical_update_fmapg(pi_old, eta, adv, pg_method):
    if pg_method == 'sPPO':
        # (refer Sec. 4.2 Tabular Parameterization
        # https://arxiv.org/pdf/2108.05828.pdf)
        pi_new = pi_old * np.maximum(1 + eta * adv, 0)  # exact FMA-PG update
        pi_new = pi_new / pi_new.sum(1).reshape(-1, 1) # normalize
    elif pg_method == 'MDPO':
        # updates the MDPO policy using the exact FMA-PG update
        # (for instance, see Eq. 2, https://arxiv.org/pdf/2005.09814.pdf)
        pi_new = pi_old * np.exp(eta * adv) # exact FMA-PG update
        pi_new = pi_new / pi_new.sum(1).reshape(-1, 1) # normalize 
    else:
        raise NotImplementedError()
    return pi_new

# computes the true gradient of the FMA-PG loss wrt policy parameters omega
# again refer to the "notes_fmapg.pdf" document
def calc_grad_fmapg(omega, pi_t, adv_t, dpi_t, eta, pg_method):
    pi = softmax(omega)
    if pg_method == 'sPPO':
        grad = dpi_t.reshape(-1, 1) * (pi_t * (adv_t + 1 / eta) - pi / eta)
    elif pg_method == 'MDPO':
        KL = (pi * np.log(pi / pi_t)).sum(1).reshape(-1, 1)
        adv_sum = (pi * adv_t).sum(1).reshape(-1, 1)
        grad = (dpi_t.reshape(-1, 1) / eta) * pi \
            * (eta * adv_t - eta * adv_sum - np.log(pi / pi_t) + KL)
    else:
        raise NotImplementedError()
    return grad

# for computing v^pi = (I - gamma P_pi)^{-1} r_pi
# P_pi(s' | s) = sum_a P(s' | s, a) * pi(a | s)
# environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
# policy pi: (i, j)th element = Pr{A = a_j | S = s_i}
def calc_vpi(env, pi, FLAG_V_S0=False):
    p_pi = np.einsum('xay,xa->xy', env.P, pi)
    r_pi = np.einsum('xa,xa->x', env.R, pi)
    v_pi = np.linalg.solve(np.eye(env.state_space) - env.gamma * p_pi, r_pi)

    if FLAG_V_S0: # calculate v_s0
        v_s0 = np.dot(env.p0, v_pi)
        return v_s0
    else:
        return v_pi

# for computing q^pi
# using q^pi(s, a) = r(s, a) + gamma * sum_s' p(s' | s, a) * v^pi(s')
def calc_qpi(env, pi):
    v_pi = calc_vpi(env, pi)
    q_pi = env.R + gamma * np.einsum('xay,y->xa', env.P, v_pi)

    return q_pi

# computing the normalized occupancy measure d^pi
# d^pi = (1 - gamma) * mu (I - gamma P_pi)^{-1},
# where mu is the start state distribution
def calc_dpi(env, pi):
    p_pi = np.einsum('xay,xa->xy', env.P, pi)
    d_pi = (1 - env.gamma) * \
        np.linalg.solve(np.eye(env.state_space) - env.gamma * p_pi.T, env.p0)

    # for addressing numerical errors; but not really needed?
    d_pi /= d_pi.sum() 
    return d_pi

# compute q*
def calc_q_star(env, num_iters=1000):
    q = np.zeros((env.state_space, env.action_space))
    for i in range(num_iters):
        q_star = q.max(1)
        q_new = env.R + np.einsum("xay,y->xa", env.gamma * env.P, q_star)
        q = q_new.copy()
    return q

# compute v*
def calc_v_star(env, num_iters=1000):
    v = np.zeros(env.state_space)
    for i in range(num_iters):
        v_new = (env.R + np.einsum("xay,y->xa", env.gamma * env.P, v)).max(1)
        v = v_new.copy()
    return v

def calc_pi_star(env, num_iters=1000):
    # just go greedy wrt q_star
    q_star = calc_q_star(env=env, num_iters=num_iters)
    pi_star = np.zeros((env.state_space, env.action_space))
    pi_star[range(env.state_space), q_star.argmax(1)] = 1
    
    return pi_star

def run_experiment_approx(env, gamma, pg_method, num_iters, eta,
                          num_inner_updates, alpha):
    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct representation)
    theta = init_theta(num_states, num_actions, theta_init=0)
    pi = softmax(theta)
    
    # evaluate the policies
    vpi_list_outer = [calc_vpi(env, pi, FLAG_V_S0=True)]
    vpi_list_inner = []

    # learning loop
    for T in range(num_iters):
        print(T)
        adv = calc_qpi(env, pi) - calc_vpi(env, pi).reshape(-1, 1)
        dpi = calc_dpi(env, pi)

        # where would have the exact update landed?
        exact_new_pi = analytical_update_fmapg(pi, eta, adv, pg_method)
        vpi_list_outer.append(calc_vpi(env, exact_new_pi, FLAG_V_S0=True))

        # gradient based update
        tmp_list = [calc_vpi(env, pi, FLAG_V_S0=True)]
        omega = theta.copy()
        dist_from_pi_new = [np.linalg.norm(pi - exact_new_pi)]
        for k in range(num_inner_updates):
            # do one gradient ascent step
            grad = calc_grad_fmapg(omega=omega, pi_t=pi, adv_t=adv,
                                   dpi_t=dpi, eta=eta, pg_method=pg_method)
            omega = omega + alpha * grad

            # save the optimization objective
            pi_tmp = softmax(omega)
            tmp_list.append(calc_vpi(env, pi_tmp, FLAG_V_S0=True))
            dist_from_pi_new.append(np.linalg.norm(pi_tmp - exact_new_pi))
            
        # plt.plot(dist_from_pi_new)
        # plt.show()

        vpi_list_inner.append(tmp_list)

        # update the policy to the approximate new point
        theta = omega.copy()
        pi = softmax(theta)

    return np.array(vpi_list_inner), np.array(vpi_list_outer)
 
def run_experiment_exact(num_iters, gamma, eta):
    # create the environment
    env = cliff_gw(gamma=gamma)
    num_states = env.state_space
    num_actions = env.action_space

    # estimate the optimal policy
    v_star = np.dot(calc_v_star(env), env.p0)
    pi_star = calc_pi_star(env)

    vpi_dict = {'MDPO': [], 'sPPO': []}
    for pg_method in ['MDPO', 'sPPO']:
        # initialize pi (uniform policy with direct representation)
        pi = generate_uniform_policy(num_states, num_actions)

        # evaluate the policies
        vpi_dict[pg_method].append(calc_vpi(env, pi, FLAG_V_S0=True))

        # learning loop
        for T in range(num_iters):
            adv = calc_qpi(env, pi) - calc_vpi(env, pi).reshape(-1, 1)
            pi = analytical_update_fmapg(pi, eta, adv, pg_method) # update pi

            # evaluate the policies    
            vpi_dict[pg_method].append(calc_vpi(env, pi, FLAG_V_S0=True))

    # generate the plots
    # fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    # axs[0].set_title(r'$\pi*$' + ' policy visualization')
    # plot_grid(axs[0], xlim=7, ylim=7)
    # plot_policy(axs[0], pi_star, xlim=7, ylim=7)

    # axs[1].set_title('sPPO')
    # plot_grid(axs[1], xlim=7, ylim=7)
    # plot_policy(axs[1], pi_sPPO, xlim=7, ylim=7)

    # axs[2].set_title('MDPO')
    # plot_grid(axs[2], xlim=7, ylim=7)
    # plot_policy(axs[2], pi_MDPO, xlim=7, ylim=7)

    # axs[3].set_title('Learning Curves')
    # axs[3].plot(vpi_dict['MDPO'], color='tab:red', label='MDPO')
    # axs[3].plot(vpi_dict['sPPO'], color='tab:blue', label='sPPO')
    # axs[3].plot(range(num_iters + 1), [v_star] * (num_iters + 1), 'k--',
    #             label='v*')  

    # axs[0].legend()
    # axs[3].legend()
    
    # axs[3].set_xlabel(r'$v^\pi(s_0)$' + ' vs Timesteps')
    # axs[3].set_ylim([0, v_star + 0.1])

    # for i in range(3):
    #     axs[i].axis('off')
    # axs[3].spines['top'].set_visible(False)
    # axs[3].spines['right'].set_visible(False)

    # plt.savefig('numIters{}__eta{}__gamma{}.jpg'.format(num_iters, eta, gamma))
    # plt.close()

    return vpi_dict

    
#======================================================================
# actual learning code
#======================================================================
gamma_list = [0.9] # [0.1, 0.5, 0.9, 0.95, 0.99]
num_steps = 200000
eta_list = [0.03] #, 1]
pg_method = 'MDPO'
num_inner_updates_list = [100, 50, 10, 1]
alpha = 0.35

#======================================================================
# learning curves against the number of iterations (different steps)
#======================================================================
# fig, axs = plt.subplots(1, 1)

# for gamma in gamma_list:
#     env = cliff_gw(gamma=gamma)
#     v_star = np.dot(calc_v_star(env), env.p0) # evaluate the optimal policy

#     for eta in eta_list:
#         # analytical FMA-PG
#         vpi_analytical_dict = run_experiment_exact(
#             num_iters=num_steps, gamma=gamma, eta=eta)
#         plt.plot(vpi_analytical_dict[pg_method], label='analytical')
        
#         for num_inner_updates in num_inner_updates_list:

#             # gradient based FMA-PG
#             vpi_list_inner, vpi_list_outer = run_experiment_approx(
#                 env=env, pg_method=pg_method, gamma=gamma,  num_iters=num_steps,
#                 eta=eta, num_inner_updates=num_inner_updates, alpha=alpha)
#             plt.plot(vpi_list_inner[:, -1],
#                      label='FMAPG_m:{}'.format(num_inner_updates))
# axs.set_ylim([0, v_star + 0.1])
# plt.legend()
# # plt.show()
# plt.savefig('learning_curves_against_iters.pdf')
# plt.close()

#======================================================================
# learning curves against the number of update steps
#======================================================================
fig, axs = plt.subplots(1, 1)

for gamma in gamma_list:
    env = cliff_gw(gamma=gamma)
    v_star = np.dot(calc_v_star(env), env.p0) # evaluate the optimal policy

    for eta in eta_list:
        # analytical FMA-PG
        vpi_analytical_dict = run_experiment_exact(
            num_iters=num_steps, gamma=gamma, eta=eta)
        plt.plot(vpi_analytical_dict[pg_method], label='analytical')
        
        for num_inner_updates in num_inner_updates_list:
            num_iters = int(num_steps / num_inner_updates)
            
            # gradient based FMA-PG
            vpi_list_inner, vpi_list_outer = run_experiment_approx(
                env=env, pg_method=pg_method, gamma=gamma,  num_iters=num_iters,
                eta=eta, num_inner_updates=num_inner_updates, alpha=alpha)
            plt.plot(vpi_list_inner[:, 0:-1].flatten(), # vpi_list_inner[:, -1],
                     label='FMAPG_m:{}'.format(num_inner_updates))
axs.set_ylim([0, v_star + 0.1])
plt.legend()
plt.savefig('learning_curves_against_steps.pdf')
   
    
#----------------------------------------------------------------------
# testing code
#----------------------------------------------------------------------
# v_star = np.dot(calc_v_star(env), env.p0)
# pi_star = calc_pi_star(env)
# v2 = calc_vpi(env, pi_star, FLAG_V_S0=True)

# fig, ax = plt.subplots(1, 1)
# plot_grid(ax, xlim=7, ylim=7)
# plot_policy(ax, pi_star, xlim=7, ylim=7)
# plt.axis('equal')
# plt.show()

# pdb.set_trace()
