import numpy as np
import matplotlib.pyplot as plt

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
def take_action(env, pi, state):
    action = np.random.choice(env.action_space, p=pi[state])
    action_prob = pi[state, action]
    return action, action_prob

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

def estimate_advantage(env, pi, num_traj, alg, qpi_old=None, stepsize=None):
    if alg == 'monte_carlo_avg':
        qpi = np.zeros((env.state_space, env.action_space))
        cnt = np.zeros((env.state_space, env.action_space))
    elif alg == 'sarsa':
        qpi = qpi_old.copy()
    else:
        raise NotImplementedError()
    
    qpi_true = env.calc_qpi(pi)
    error_list = [np.linalg.norm(qpi - qpi_true)]
    
    for traj_i in range(num_traj):
        #========================================================
        # sample a trajectory
        #--------------------------------------------------------
        traj = {'state_list': [],
                'action_list': [],
                'action_prob_list': [],
                'reward_list': [],
                'next_state_list': []}
        state = env.reset()
        done = 'false'
        while done == 'false':
            action, action_prob = take_action(env, pi, state)
            next_state, reward, done, _ = env.step(action)

            traj['state_list'].append(state)
            traj['action_list'].append(action)
            traj['action_prob_list'].append(action_prob)
            traj['reward_list'].append(reward)
            traj['next_state_list'].append(next_state)

            state = next_state
        traj['done_status'] = done
        #========================================================

        if alg == 'monte_carlo_avg':
            G = 0
            for i in range(len(traj['state_list']) - 1, -1, -1):
                state = traj['state_list'][i]
                action = traj['action_list'][i]
                reward = traj['reward_list'][i]

                G = reward + env.gamma * G

                cnt[state, action] += 1
                qpi[state, action] = qpi[state, action] \
                    + (G - qpi[state, action]) / cnt[state, action]
        elif alg == 'sarsa':
            for i in range(len(traj['state_list']) - 1):
                state = traj['state_list'][i]
                action = traj['action_list'][i]
                reward = traj['reward_list'][i]
                next_state = traj['next_state_list'][i]

                # very last state in the trajectory
                if i == len(traj['state_list']) - 1:
                    if traj['done_status'] == 'terminal':
                        q_next_state_action = 0
                    elif traj['done_status'] == 'cutoff':
                        next_action = take_action(env, pi, next_state)
                        q_next_state_action = qpi[next_state][next_action]
                else:
                    next_action = traj['action_list'][i+1]
                    q_next_state_action = qpi[next_state][next_action]
                    
                target = reward + env.gamma * q_next_state_action
                qpi[state][action] = qpi[state][action] \
                    + stepsize * (target - qpi[state][action])

        error_list.append(np.linalg.norm(qpi - qpi_true))
        
    # pdb.set_trace()
    # plt.show(); plt.figure(); plt.plot(error_list); plt.show()
        
    vpi = (qpi * pi).sum(1).reshape(-1, 1)
    adv = qpi - vpi

    return adv, qpi.copy()

#----------------------------------------------------------------------
# functions for running full experiments
#----------------------------------------------------------------------

def run_experiment_approx(env, gamma, pg_method, num_iters, eta,
                          num_inner_updates, alpha,
                          FLAG_USE_TRUE_ADVANTAGE=True, num_traj_est_adv=10,
                          adv_est_alg='monte_carlo_avg', adv_est_stepsize=None):
    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct representation)
    theta = init_theta(num_states, num_actions, theta_init=0)
    pi = softmax(theta)
    
    # evaluate the policies
    vpi_list_outer = [env.calc_vpi(pi, FLAG_V_S0=True)]
    vpi_list_inner = []

    # learning loop
    qpi_est = np.zeros((env.state_space, env.action_space))
    for T in range(num_iters):
        if FLAG_USE_TRUE_ADVANTAGE:
            adv = env.calc_qpi(pi) - env.calc_vpi(pi).reshape(-1, 1)
        else:
            adv, qpi_est = estimate_advantage(
                env, pi=pi, num_traj=num_traj_est_adv,
                alg=adv_est_alg, qpi_old=qpi_est, stepsize=adv_est_stepsize)
        dpi = env.calc_dpi(pi)

        # where would have the exact update landed?
        exact_new_pi = analytical_update_fmapg(pi, eta, adv, pg_method)
        vpi_list_outer.append(env.calc_vpi(exact_new_pi, FLAG_V_S0=True))

        # gradient based update
        tmp_list = [env.calc_vpi(pi, FLAG_V_S0=True)]
        omega = theta.copy()
        dist_from_pi_new = [np.linalg.norm(pi - exact_new_pi)]
        for k in range(num_inner_updates):
            # do one gradient ascent step
            grad = calc_grad_fmapg(omega=omega, pi_t=pi, adv_t=adv,
                                   dpi_t=dpi, eta=eta, pg_method=pg_method)
            omega = omega + alpha * grad

            # save the optimization objective
            pi_tmp = softmax(omega)
            tmp_list.append(env.calc_vpi(pi_tmp, FLAG_V_S0=True))
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
    env = CliffWorld()
    num_states = env.state_space
    num_actions = env.action_space

    # estimate the optimal policy
    v_star = np.dot(env.calc_v_star(), env.mu)
    pi_star = env.calc_pi_star()

    vpi_dict = {'MDPO': [], 'sPPO': []}
    for pg_method in ['MDPO', 'sPPO']:
        # initialize pi (uniform policy with direct representation)
        pi = generate_uniform_policy(num_states, num_actions)

        # evaluate the policies
        vpi_dict[pg_method].append(env.calc_vpi(pi, FLAG_V_S0=True))

        # learning loop
        for T in range(num_iters):
            adv = env.calc_qpi(pi) - env.calc_vpi(pi).reshape(-1, 1)
            pi = analytical_update_fmapg(pi, eta, adv, pg_method) # update pi

            # evaluate the policies    
            vpi_dict[pg_method].append(env.calc_vpi(pi, FLAG_V_S0=True))

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
