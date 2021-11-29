import numpy as np
from scipy.stats import entropy
# import matplotlib.pyplot as plt
import pdb

#----------------------------------------------------------------------
# plotting functions
#----------------------------------------------------------------------
def plot_grid(ax, xlim=4, ylim=5):
    x_center = np.arange(xlim) + 0.5
    y_center = np.arange(ylim) + 0.5

    for x in x_center:
        for y in y_center:
            if int((x - 0.5) * ylim + (y - 0.5)) == 0:
                ax.scatter(x, y, s=1500, color='tab:blue', marker='s',
                           label='start', alpha=0.5)
            elif int((x - 0.5) * ylim + (y - 0.5)) in [1, 2, 3]:
                ax.scatter(x, y, s=1500, color='tab:red', marker='x',
                           label='chasm', alpha=0.5)
            elif int((x - 0.5) * ylim + (y - 0.5)) == 4:
                ax.scatter(x, y, s=1500, color='tab:green', marker='s',
                           label='end', alpha=0.5)
            else:
                ax.scatter(x, y, s=0.1, color='purple')

    for x in range(xlim + 1):
        ax.plot((x, x), (0, ylim), linewidth=0.5, color='black')

    for y in range(ylim + 1):
        ax.plot((0, xlim), (y, y), linewidth=0.5, color='black')

       
def plot_policy(ax, pi, xlim=4, ylim=5):
    for s_x in range(xlim):
        for s_y in range(ylim):
            x_center = s_x + 0.5
            y_center = s_y + 0.5
            s = s_x * ylim + s_y
            diff = pi[s] / 2

            # down
            ax.quiver(x_center, y_center, 0, -diff[0],
                      color='tab:orange', width=0.013, headwidth=2,
                      headlength=4, scale=1, scale_units='xy', linewidth=1)
            # up
            ax.quiver(x_center, y_center, 0, +diff[1],
                      color='tab:blue', width=0.013, headwidth=2, 
                      headlength=4, scale=1, scale_units='xy', linewidth=1)
            # left
            ax.quiver(x_center, y_center, -diff[2], 0,
                      color='tab:red', width=0.013, headwidth=2,
                      headlength=4, scale=1, scale_units='xy', linewidth=1)
            # right
            ax.quiver(x_center, y_center, +diff[3], 0,
                      color='tab:green', width=0.013, headwidth=2, 
                      headlength=4, scale=1, scale_units='xy', linewidth=1)

#----------------------------------------------------------------------
# utility functions
#----------------------------------------------------------------------
def take_action(env, pi, state):
    action = np.random.choice(env.action_space, p=pi[state])
    action_prob = pi[state, action]
    return action, action_prob

def softmax(x):
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    out = e_x / e_x.sum(1).reshape(-1, 1)
    # assert np.allclose(out.sum(1), np.ones(21))
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
def calc_grad_fmapg(omega, pi_t, adv_t, dpi_t, eta, pg_method,
                    version_grad='numerically_stable'):
    pi = softmax(omega)
    if pg_method == 'sPPO':
        grad = dpi_t.reshape(-1, 1) * (pi_t * (adv_t + 1 / eta) - pi / eta)
    elif pg_method == 'MDPO':
        if version_grad == 'naive':
            KL = (pi * np.log(pi / pi_t)).sum(1).reshape(-1, 1)
            adv_sum = (pi * adv_t).sum(1).reshape(-1, 1)
            grad = (dpi_t.reshape(-1, 1) / eta) * pi \
                * (eta * adv_t - eta * adv_sum - np.log(pi / pi_t) + KL)
        elif version_grad == 'numerically_stable':
            # for numerical stability, use the following:
            KL2 = entropy(pi, pi_t, axis=1).reshape(-1, 1)
            clipped_pi_t = np.clip(pi_t, a_min=1e-6, a_max=1 - 1e-6)
            grad = (dpi_t.reshape(-1, 1) / eta) * pi \
                * (eta * adv_t - eta * adv_sum - np.log(pi / clipped_pi_t) + KL2)
    else:
        raise NotImplementedError()
    return grad

#----------------------------------------------------------------------
# TRPO and PPO
#----------------------------------------------------------------------
# returns the A matrix. It is an SA x SA matrix computed using equations given
# in Section 3 of the notes
def calc_A_matrix(num_states, num_actions, dpi_t, pi):
    n = num_states * num_actions
    A = np.zeros((n, n))
    for state_i in range(num_states):
        pi_s = pi[state_i, :]
        beg = num_actions * state_i
        end = beg + num_actions
        A[beg:end, beg:end] \
            = dpi_t[state_i] * (np.diag(pi_s) - np.outer(pi_s, pi_s))
    return A

def calc_A_matrix_slow(num_states, num_actions, dpi_t, pi):
    # only for verifying if calc_A_matrix() works correctly
    n = num_states * num_actions
    A = np.zeros((n, n))
    for state_i in range(num_states):
        for action_i in range(num_actions):
            for state_j in range(num_states):
                for action_j in range(num_actions):
                    row = state_i * num_actions + action_i
                    col = state_j * num_actions + action_j
                    A[row, col] = (state_i == state_j) * dpi_t[state_j] \
                        * ((action_i == action_j) - pi[state_j, action_i]) \
                        * pi[state_j, action_j]
    return A

# this is the TRPO grad vector, but stored as an S x A matrix,
# again computed using equations given in Section 3 of the notes
def calc_grad_vector(dpi_t, qpi_t, pi):
    grad = dpi_t.reshape(-1, 1) * pi \
        * (qpi_t - (pi * qpi_t).sum(1).reshape(-1, 1))
    return grad

# this calculates the update direction s = A^{-1} grad. We first flatten grad,
# calculate s using this equation, and then reshape s again into an
# S x A matrix
def calc_update_direction(grad, A):
    grad_flatten = grad.reshape(-1, 1)
    # s_flatten = np.linalg.solve(A, grad_flatten)
    # Having a lot of problem with A being singular. So using pseudo-inverse
    update_direction_flatten = np.matmul(np.linalg.pinv(A), grad_flatten)    
    update_direction = update_direction_flatten.reshape(grad.shape)
    return update_direction

# compute the maximum stepsize beta using the equations given in Appendix C of
# Schulman et al. (2015)
def calc_max_stepsize_beta(update_direction, A, delta):
    update_direction_flatten = update_direction.reshape(-1, 1)
    beta = np.sqrt(2 * delta \
                   / np.matmul(np.matmul(update_direction_flatten.T, A),
                               update_direction_flatten).item())
    return beta

def compute_trpo_loss(dpi_t, qpi_t, pi):
    J = (dpi_t * (pi * qpi_t).sum(1)).sum()
    return J

def compute_trpo_constraint(dpi_t, pi_t, pi):
    # C = (dpi_t * (pi_t * np.log(pi_t / pi)).sum(1)).sum()
    # an equivalent, but better, way that avoids NaN values from 0/0 division
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    C = (dpi_t * entropy(pi_t, pi, axis=1)).sum()
    return C

def trpo_update(omega, pi_t, dpi_t, qpi_t, delta, decay_factor,
                num_states, num_actions):
    pi = softmax(omega)
    J_old = compute_trpo_loss(dpi_t=dpi_t, qpi_t=qpi_t, pi=pi)
    A = calc_A_matrix(
        num_states=num_states, num_actions=num_actions, dpi_t=dpi_t, pi=pi)
    grad = calc_grad_vector(dpi_t=dpi_t, qpi_t=qpi_t, pi=pi)
    # A2 = calc_A_matrix_slow(
    #     num_states=num_states, num_actions=num_actions, dpi_t=dpi_t, pi=pi)
    # assert np.allclose(A, A2)
    update_direction = calc_update_direction(grad=grad, A=A) # grad
    if np.allclose(np.zeros(update_direction.shape), update_direction):
        beta = 0
    else:
        beta = calc_max_stepsize_beta(update_direction=update_direction,
                                      A=A, delta=delta)
    while True: # backtracking line search
        omega_tmp = omega + beta * update_direction
        pi_tmp = softmax(omega_tmp)
        J_tmp = compute_trpo_loss(dpi_t=dpi_t, qpi_t=qpi_t, pi=pi_tmp)
        C_tmp = compute_trpo_constraint(dpi_t=dpi_t, pi_t=pi_t, pi=pi_tmp)
        if C_tmp <= delta and J_tmp >= J_old:
            break
        beta = decay_factor * beta
        
    return omega_tmp

def trpo_kl_ls_update(omega, pi_t, dpi_t, qpi_t, decay_factor, zeta, 
                      armijo_constant=0, max_backtracking_iters=None,
                      warm_start_beta_init=10, warm_start_beta_factor=10):
    pi = softmax(omega)
    J_old = compute_trpo_loss(dpi_t=dpi_t, qpi_t=qpi_t, pi=pi) \
        - zeta * compute_trpo_constraint(dpi_t=dpi_t, pi_t=pi_t, pi=pi)
    grad = calc_grad_trpo_kl(omega=omega, pi_t=pi_t, dpi_t=dpi_t, qpi_t=qpi_t,
                             zeta=zeta)

    if np.allclose(np.zeros(grad.shape), grad):# or np.linalg.norm(grad) < 1e-8:
        beta = 0
        return omega, beta
    else:
        beta = warm_start_beta_factor * warm_start_beta_init

    # end after finite number of iterations (say 100)
    backtracking_iter = 0
    while True: # backtracking line search
        omega_tmp = omega + beta * grad
        pi_tmp = softmax(omega_tmp)
        J_tmp = compute_trpo_loss(dpi_t=dpi_t, qpi_t=qpi_t, pi=pi_tmp) \
            - zeta * compute_trpo_constraint(dpi_t=dpi_t, pi_t=pi_t, pi=pi_tmp)

        # print(beta, J_tmp - J_old)
        # if np.isnan(J_tmp - J_old):
        #     pdb.set_trace()
        # Armijo's line search condition
        if J_tmp >= J_old + armijo_constant * beta * np.linalg.norm(grad)**2:
            break

        # if too many backtracking steps, just make a super small increment
        if max_backtracking_iters is not None:
            if backtracking_iter > max_backtracking_iters:
                omega_tmp = omega + 1e-8 * grad
                break

        beta = decay_factor * beta
        backtracking_iter += 1
        
    # print values of beta at each run
    # print('beta', beta)
    return omega_tmp, beta
    
def calc_grad_ppo(omega, pi_t, adv_t, dpi_t, epsilon):
    pi = softmax(omega)
    ratio = pi / pi_t
    cond = np.logical_or(np.logical_and(adv_t > 0, ratio < 1 + epsilon),
                         np.logical_and(adv_t < 0, ratio > 1 - epsilon))

    grad = dpi_t.reshape(-1, 1) * pi \
        * (cond * adv_t - (pi * cond * adv_t).sum(1).reshape(-1, 1))
    return grad

def calc_grad_trpo_kl(omega, pi_t, dpi_t, qpi_t, zeta):
    pi = softmax(omega)
    
    grad_t1 = dpi_t.reshape(-1, 1) * pi \
        * (qpi_t - (pi * qpi_t).sum(1).reshape(-1, 1))
    grad_t2 = zeta * dpi_t.reshape(-1, 1) * (pi - pi_t)
    grad = grad_t1 + grad_t2
    
    return grad

#----------------------------------------------------------------------
# advantage estimation
#----------------------------------------------------------------------
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
        
    # plt.show(); plt.figure(); plt.plot(error_list); plt.show()
        
    vpi = (qpi * pi).sum(1).reshape(-1, 1)
    adv = qpi - vpi

    return adv, qpi.copy()

#----------------------------------------------------------------------
# function for running full experiments
#----------------------------------------------------------------------
def run_experiment(env, pg_method, num_iters, eta, delta, decay_factor, epsilon,
                   FLAG_ANALYTICAL_GRADIENT, num_inner_updates, alpha,
                   FLAG_TRUE_ADVANTAGE, adv_estimate_alg,
                   num_traj_estimate_adv, adv_estimate_stepsize,
                   FLAG_SAVE_INNER_STEPS, zeta, armijo_constant,
                   max_backtracking_iters,
                   FLAG_BETA_WARM_START,
                   warm_start_beta_init, warm_start_beta_factor):    
    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct representation)
    theta = init_theta(num_states, num_actions, theta_init=0)
    pi = softmax(theta)
    
    # evaluate the policies
    vpi_list_outer = [env.calc_vpi(pi, FLAG_RETURN_V_S0=True)]
    vpi_list_inner = [] if FLAG_SAVE_INNER_STEPS else None

    if not FLAG_TRUE_ADVANTAGE:
        qpi_estimate = np.zeros((env.state_space, env.action_space))

    # learning loop
    for T in range(num_iters):
        # print(T)
        dpi = env.calc_dpi(pi)
        
        if FLAG_TRUE_ADVANTAGE:
            qpi = env.calc_qpi(pi)
            adv = qpi - env.calc_vpi(pi).reshape(-1, 1)
        else:
            adv, qpi_estimate = estimate_advantage(
                env, pi=pi, num_traj=num_traj_estimate_adv,
                alg=adv_estimate_alg, qpi_old=qpi_estimate,
                stepsize=adv_estimate_stepsize)
            qpi = qpi_estimate.copy()

        # gradient based update
        vpi_list_outer.append(env.calc_vpi(pi, FLAG_RETURN_V_S0=True))

        if FLAG_ANALYTICAL_GRADIENT: # only for sPPO and MDPO
            pi = analytical_update_fmapg(pi, eta, adv, pg_method)
        else:
            if FLAG_SAVE_INNER_STEPS:
                tmp_list = [env.calc_vpi(pi, FLAG_RETURN_V_S0=True)]
                
            omega = theta.copy()
            warm_start_beta_init__current_k = warm_start_beta_init
            for k in range(num_inner_updates):
                # do one gradient ascent step
                if pg_method in ['sPPO', 'MDPO']:
                    grad = calc_grad_fmapg(
                        omega=omega, pi_t=pi, adv_t=adv, dpi_t=dpi, eta=eta,
                        pg_method=pg_method)
                    omega = omega + alpha * grad                    
                elif pg_method == 'TRPO':
                    omega_tmp = trpo_update(
                        omega=omega, pi_t=pi, dpi_t=dpi, qpi_t=qpi, delta=delta,
                        decay_factor=decay_factor, num_states=num_states,
                        num_actions=num_actions)
                    omega = omega_tmp.copy()
                elif pg_method == 'PPO':
                    grad = calc_grad_ppo(
                        omega=omega, pi_t=pi, adv_t=adv, dpi_t=dpi,
                        epsilon=epsilon)
                    omega = omega + alpha * grad
                elif pg_method == 'TRPO_KL':
                    grad = calc_grad_trpo_kl(
                        omega=omega, pi_t=pi, dpi_t=dpi, qpi_t=qpi, zeta=zeta)
                    omega = omega + alpha * grad
                elif pg_method == 'TRPO_KL_LS':
                    omega_tmp, beta_init_tmp = \
                        trpo_kl_ls_update(
                            omega=omega, pi_t=pi, dpi_t=dpi, qpi_t=qpi,
                            decay_factor=decay_factor, zeta=zeta,
                            armijo_constant=armijo_constant,
                            max_backtracking_iters=max_backtracking_iters,
                            warm_start_beta_init=warm_start_beta_init__current_k,
                            warm_start_beta_factor=warm_start_beta_factor)
                    omega = omega_tmp.copy()
                    if FLAG_BETA_WARM_START:
                        warm_start_beta_init = beta_init_tmp
                else:
                    raise NotImplementedError()
                    
                # save the optimization objective
                if FLAG_SAVE_INNER_STEPS:
                    pi_tmp = softmax(omega)
                    tmp_list.append(env.calc_vpi(pi_tmp, FLAG_RETURN_V_S0=True))

            if FLAG_SAVE_INNER_STEPS:
                vpi_list_inner.append(tmp_list)

            # update the policy to the new approximate point
            theta = omega.copy()
            pi = softmax(theta)

    vpi_list_outer = np.array(vpi_list_outer)
    if FLAG_SAVE_INNER_STEPS:
        vpi_list_inner = np.array(vpi_list_inner)
        
    return vpi_list_outer, vpi_list_inner, pi

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

# color_dict = {100: 'tab:red', 50: 'tab:green', 10: 'tab:blue'}
# linestyle_dict = {100: '--', 50: '-.', 10: ':'}
