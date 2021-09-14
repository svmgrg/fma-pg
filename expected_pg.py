import numpy as np
import matplotlib.pyplot as plt
import pdb

from src.envs.gridworld_mdp import cliff_gw

# only runs when put inside "commented_code" file 

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
# generate a uniformly random tabular policy (direct representation)
def generate_uniform_policy(num_states, num_actions):
    return np.ones((num_states, num_actions)) / num_actions

# updates the softmax_ppo policy using the exact FMA-PG update
# (refer Sec. 4.2 Tabular Parameterization, https://arxiv.org/pdf/2108.05828.pdf)
def update_sPPO(pi_old, eta, adv):
    pi_new = pi_old * np.maximum(1 + eta * adv, 0)  # exact FMA-PG update
    pi_new = pi_new / pi_new.sum(1).reshape(-1, 1) # normalize (projection step)
    return pi_new

# updates the MDPO policy using the exact FMA-PG update
# (for instance, see Eq. 2, https://arxiv.org/pdf/2005.09814.pdf)
def update_MDPO(pi_old, eta, adv):
    pi_new = pi_old * np.exp(eta * pi_old * adv) # exact FMA-PG update
    pi_new = pi_new / pi_new.sum(1).reshape(-1, 1) # normalize (projection step)
    return pi_new

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
    # d_pi /= d_pi.sum() 
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

def run_experiment(num_iters, gamma, eta):
    # create the environment
    env = cliff_gw(gamma=gamma)
    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct representation)
    pi_MDPO = generate_uniform_policy(num_states, num_actions)
    pi_sPPO = generate_uniform_policy(num_states, num_actions)

    # evaluate the policies
    vpi_list = {'MDPO': [], 'sPPO': []}
    vpi_list['MDPO'].append(calc_vpi(env, pi_MDPO, FLAG_V_S0=True))
    vpi_list['sPPO'].append(calc_vpi(env, pi_sPPO, FLAG_V_S0=True))

    # learning loop
    for T in range(num_iters):
        # update policies
        adv_MDPO = calc_qpi(env, pi_MDPO) - calc_vpi(env, pi_MDPO).reshape(-1, 1)
        pi_MDPO = update_MDPO(pi_MDPO, eta, adv_MDPO)

        adv_sPPO = calc_qpi(env, pi_sPPO) - calc_vpi(env, pi_sPPO).reshape(-1, 1)
        pi_sPPO = update_sPPO(pi_sPPO, eta, adv_sPPO)

        # evaluate the policies    
        vpi_list['MDPO'].append(calc_vpi(env, pi_MDPO, FLAG_V_S0=True))
        vpi_list['sPPO'].append(calc_vpi(env, pi_sPPO, FLAG_V_S0=True))

        # print('iteration #{}'.format(T))

    # estimate the optimal policy
    v_star = np.dot(calc_v_star(env), env.p0)
    pi_star = calc_pi_star(env)
    v2 = calc_vpi(env, pi_star, FLAG_V_S0=True)

    # generate the plots
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].set_title(r'$\pi*$' + ' policy visualization')
    plot_grid(axs[0], xlim=7, ylim=7)
    plot_policy(axs[0], pi_star, xlim=7, ylim=7)
    

    axs[1].set_title('sPPO')
    plot_grid(axs[1], xlim=7, ylim=7)
    plot_policy(axs[1], pi_sPPO, xlim=7, ylim=7)

    axs[2].set_title('MDPO')
    plot_grid(axs[2], xlim=7, ylim=7)
    plot_policy(axs[2], pi_MDPO, xlim=7, ylim=7)

    axs[3].set_title('Learning Curves')
    axs[3].plot(vpi_list['MDPO'], color='tab:red', label='MDPO')
    axs[3].plot(vpi_list['sPPO'], color='tab:blue', label='sPPO')
    axs[3].plot(range(num_iters + 1), [v_star] * (num_iters + 1), 'k--',
                label='v*')  

    axs[0].legend()
    axs[3].legend()
    
    axs[3].set_xlabel(r'$v^\pi(s_0)$' + ' vs Timesteps')
    axs[3].set_ylim([0, v_star + 0.1])

    for i in range(3):
        axs[i].axis('off')
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)

    plt.savefig('numIters{}__eta{}__gamma{}.jpg'.format(num_iters, eta, gamma))
    plt.close()

    
#----------------------------------------------------------------------
# actual learning code
#----------------------------------------------------------------------
gamma_list = [0.9] # [0.1, 0.5, 0.9, 0.95, 0.99]
num_iters = 200000
eta_list = [0.03, 1]

for eta in eta_list:
    for gamma in gamma_list:
        run_experiment(num_iters=num_iters, gamma=gamma, eta=eta)
        print(eta, gamma)
    
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
