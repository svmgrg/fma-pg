import numpy as np
from scipy.stats import entropy
import pdb

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
    return out

# generate a uniformly random tabular policy (direct representation)
def generate_uniform_policy(num_states, num_actions):
    return np.ones((num_states, num_actions)) / num_actions

# generate action preferences (for softmax policy representation)
def init_theta(num_states, num_actions, theta_init=0):
    return theta_init * np.ones((num_states, num_actions))

def make_single_inner_loop_update(optim_type, stepsize_type, omega,
                                  update_direction, calc_objective_fn,
                                  calc_constraint_fn=None,
                                  objective_grad=None, eta=None,
                                  epsilon=None, delta=None, alpha_fixed=0.1,
                                  alpha_init=100, decay_factor=0.1,
                                  armijo_const=0, max_backtracking_steps=100):
    '''
    omega is the initial weight vector for the policy, 
    update_direction is equal to the gradient by default or can be the
                     direction of the steepest descent as in TRPO,
    calc_objective_fn() is used to calculate the objective value, 
    calc_constraint_fn() is used to check whether the constraint is satisfied.
    '''
    assert optim_type in ['regularized', 'constrained']
    assert stepsize_type in ['fixed_stepsize', 'line_search']

    if stepsize_type == 'fixed_stepsize':
        if optim_type == 'regularized':
            omega_tmp = omega + alpha_fixed * update_direction
        elif optim_type == 'constrained':
            raise NotImplementedError()
    elif stepsize_type == 'line_search':
        J_old = calc_objective_fn(omega)
        # the term cm comes from
        # https://en.wikipedia.org/wiki/Backtracking_line_search#Motivation
        cm = armijo_const * np.dot(objective_grad, update_direction)
        alpha = alpha_init
        t = 0
        while True:
            omega_tmp = omega + alpha * update_step
            J_tmp = calc_objective_fn(omega_tmp)
            C_tmp = calc_constraint_fn(omega_tmp)
            if ((optim_type == 'regularized' and J_tmp >= J_old + alpha * cm)
                or
                (optim_type == 'constrained' and
                 J_tmp >= J_old + alpha * cm and C_tmp <= delta)):
                break # use this alpha
            elif t > max_backtracking_steps:
                alpha = 1e-6
                omega_tmp = omega + alpha * update_step
                break
            else:
                alpha = decay_factor * alpha
                t += 1
    
    return omega_tmp, alpha

#----------------------------------------------------------------------
# Old functions for calculating sPPO and MDPO gradients. They compute
# the true gradient of the FMA-PG loss wrt policy parameters omega.
# (Kept here as a check for the new functions)
#----------------------------------------------------------------------
def calc_grad_sPPO(omega, pi_t, adv_t, dpi_t, eta):
    pi = softmax(omega)
    grad = dpi_t.reshape(-1, 1) * (pi_t * (adv_t + 1 / eta) - pi / eta)
    return grad

def calc_grad_MDPO(omega, pi_t, adv_t, dpi_t, eta):
    pi = softmax(omega)
    KL = entropy(pi, pi_t, axis=1).reshape(-1, 1)
    adv_sum = (pi * adv_t).sum(1).reshape(-1, 1)
    clipped_pi_t = np.clip(pi_t, a_min=1e-6, a_max=1 - 1e-6)
    grad = (dpi_t.reshape(-1, 1) 1/ eta) * pi \
        * (eta * adv_t - eta * adv_sum - np.log(pi / clipped_pi_t) + KL)
    return grad

#----------------------------------------------------------------------
# Combined Functions
#----------------------------------------------------------------------
def analytical_update_fmapg(pi_old, eta, adv, pg_method):
    assert pg_method in ['sPPO', 'MDPO']
    
    if pg_method == 'sPPO':
        pi_new = pi_old * np.maximum(1 + eta * adv, 0)  # exact sPPO update
    elif pg_method == 'MDPO':
        pi_new = pi_old * np.exp(eta * adv) # exact MDPO update

    pi_new = pi_new / pi_new.sum(1).reshape(-1, 1) # normalize
    return pi_new

def calc_objective(omega, pi_t, adv_t, qpi_t, dpi_t, pg_method, epsilon=None):
    pi = softmax(omega)
    if pg_method in ['sPPO']:
        clipped_pi_t = np.clip(pi_t, a_min=1e-6, a_max=1-1e-6)        
        obj = (dpi_t * (pi_t * adv_t * np.log(pi / clipped_pi_t)).sum(1)).sum()
    elif pg_method in ['MDPO']:
        obj = (dpi_t * (pi * adv_t).sum(1)).sum()
    elif pg_method in ['TRPO']:
        obj = (dpi_t * (pi * qpi_t).sum(1)).sum()
    elif pg_method in ['PPO']:
        raise NotImplementedError()
    return obj

def calc_constraint(omega, pi_t, dpi_t, pg_method):
    pi = softmax(omega)
    if pg_method in ['sPPO', 'TRPO']:
        cst = (dpi_t * entropy(pi_t, pi, axis=1)).sum()
    elif pg_method in ['MDPO']:
        cst = (dpi_t * entropy(pi, pi_t, axis=1)).sum()
    return cst

def calc_obj_grad(omega, pi_t, adv_t, qpi_t, dpi_t, pg_method, epsilon=None):
    # note that the objective gradients for both TRPO and MDPO are exactly equal
    pi = softmax(omega)
    
    if pg_method in ['sPPO']:
        obj_grad = dpi_t.reshape(-1, 1) * pi_t * adv_t
    elif pg_method in ['TRPO']:
        qpi_exp = (pi * qpi_t).sum(1).reshape(-1, 1)
        obj_grad = dpi_t.reshape(-1, 1) * pi * (qpi_t - qpi_exp)
    elif pg_method in ['MDPO']:
        adv_exp = (pi * adv_t).sum(1).reshape(-1, 1)
        obj_grad = dpi_t.reshape(-1, 1) * pi * (adv_t - adv_exp)
    elif pg_method in ['PPO']:
        ratio = pi / pi_t
        cond = np.logical_or(np.logical_and(adv_t > 0, ratio < 1 + epsilon),
                             np.logical_and(adv_t < 0, ratio > 1 - epsilon))
        obj_grad = dpi_t.reshape(-1, 1) * pi \
            * (cond * adv_t - (pi * cond * adv_t).sum(1).reshape(-1, 1))

    return obj_grad
        
def calc_cst_grad(omega, pi_t, dpi_t, pg_method):
    pi = softmax(omega)
    if pg_method in ['sPPO', 'TRPO']:
        cst_grad =  dpi_t.reshape(-1, 1) * (pi - pi_t)
    elif pg_method in ['MDPO']:
        KL = entropy(pi, pi_t, axis=1).reshape(-1, 1)
        clipped_pi_t = np.clip(pi_t, a_min=1e-6, a_max=1-1e-6)
        cst_grad = dpi_t.reshape(-1, 1) * pi * (np.log(pi / clipped_pi_t) - KL)

    return cst_grad

#----------------------------------------------------------------------
# PPO
#----------------------------------------------------------------------
def calc_grad_ppo(omega, pi_t, adv_t, dpi_t, epsilon):
    pi = softmax(omega)
    ratio = pi / pi_t
    cond = np.logical_or(np.logical_and(adv_t > 0, ratio < 1 + epsilon),
                         np.logical_and(adv_t < 0, ratio > 1 - epsilon))

    grad = dpi_t.reshape(-1, 1) * pi \
        * (cond * adv_t - (pi * cond * adv_t).sum(1).reshape(-1, 1))
    return grad

#----------------------------------------------------------------------
# TRPO
#----------------------------------------------------------------------
# returns the A matrix: an SA x SA matrix computed using equations given
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

# calc_A_matrix_slow is an alternate way of calculating the above matrix, and
# serves as a check for the function calc_A_matrix()
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

# this calc the update direction s = A^{-1} grad. We first flatten grad,
# calculate s using this equation, and then reshape s again into an
# S x A matrix
def calc_update_direction(grad, A):
    grad_flatten = grad.reshape(-1, 1)
    # s_flatten = np.linalg.solve(A, grad_flatten)
    # had a lot of problem with A being singular --- so use pseudo-inverse
    update_direction_flatten = np.matmul(np.linalg.pinv(A), grad_flatten)    
    update_direction = update_direction_flatten.reshape(grad.shape)
    return update_direction

# compute the maximum stepsize beta using the equations given in App. C of
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

# C = (dpi_t * (pi_t * np.log(pi_t / pi)).sum(1)).sum()
# a better way that avoids NaN values from 0/0 division; see
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
def compute_trpo_constraint(dpi_t, pi_t, pi):
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
                      armijo_const=0, max_backtracking_iters=None,
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
        if J_tmp >= J_old + armijo_const * beta * np.linalg.norm(grad)**2:
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

def calc_grad_trpo_kl(omega, pi_t, dpi_t, qpi_t, zeta):
    pi = softmax(omega)
    
    grad_t1 = dpi_t.reshape(-1, 1) * pi \
        * (qpi_t - (pi * qpi_t).sum(1).reshape(-1, 1))
    grad_t2 = zeta * dpi_t.reshape(-1, 1) * (pi - pi_t)
    grad = grad_t1 + grad_t2
    
    return grad

#----------------------------------------------------------------------
# function for running full experiments
#----------------------------------------------------------------------
def run_experiment(env, pg_method, num_outer_loop_iter, num_inner_loop_iter,
                   FLAG_ANALYTICAL_GRADIENT, FLAG_SAVE_INNER_STEPS,
                   alpha_max, FLAG_WARM_START, warm_start_factor,
                   max_backtracking_iters,
                   optim_type, stepsize_type, eta, epsilon, delta,
                   alpha_fixed, decay_factor, armijo_const):
    assert pg_method in ['sPPO', 'MDPO', 'PPO', 'TRPO']
    assert optim_type in ['regularized', 'constrained']
    assert stepsize_type in ['fixed_stepsize', 'line_search']

    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct rep)
    theta = init_theta(num_states, num_actions, theta_init=0)
    pi = softmax(theta)
    
    # store the quality of the policies
    vpi_list_outer = [env.calc_vpi(pi, FLAG_RETURN_V_S0=True)]
    vpi_list_inner = [] if FLAG_SAVE_INNER_STEPS else None

    # outer learning loop
    for T in range(num_outer_loop_iter):
        dpi = env.calc_dpi(pi)
        qpi = env.calc_qpi(pi)
        adv = qpi - env.calc_vpi(pi).reshape(-1, 1)

        vpi_list_outer.append(env.calc_vpi(pi, FLAG_RETURN_V_S0=True))

        if FLAG_ANALYTICAL_GRADIENT: 
            pi = analytical_update_fmapg(pi, eta, adv, pg_method)
        else:
            # multiple gradient based updates
            if FLAG_SAVE_INNER_STEPS:
                tmp_list = [env.calc_vpi(pi, FLAG_RETURN_V_S0=True)]

            omega = theta.copy()
            used_alpha = None
                            
            # inner learning loop
            for k in range(num_inner_loop_iter): # do one grad ascent step

                if optim_type == 'regularized':
                    obj_grad = calc_obj_grad(omega=omega, pi_t=pi, adv_t=adv,
                                             dpi_t=dpi, pg_method=pg_method,
                                             epsilon=epsilon)
                    cst_grad = calc_cst_grad(omega=omega, pi_t=pi, dpi_t=dpi,
                                             pg_method=pg_method)
                    update_direction = obj_grad - (1 \ eta) * cst_grad
                    objective_grad = update_direction
                elif optim_type == 'constrained':
                    raise NotImplementedError

                if stepsize_type == 'fixed_stepsize':
                    calc_objective_fn = None
                    calc_constraint_fn = None
                    alpha_init = None
                elif stepsize_type == 'line_search':
                    calc_objective_fn = lambda omega: calc_objective(
                        omega=omega, pi_t=pi, adv_t=adv, dpi_t=dpi,
                        pg_method=pg_method, epsilon=epsilon)
                    calc_constraint_fn = lambda omega: calc_constraint(
                        omega=omega, pi_t=pi, dpi_t=dpi, pg_method=pg_method)

                    if FLAG_WARM_START and used_alpha is not None:
                        alpha_init = warm_start_factor * used_alpha
                    else:
                        alpha_init = alpha_max

                # make a single inner loop update
                updated_omega, used_alpha = make_single_inner_loop_update(
                    optim_type=optim_type, stepsize_type=stepsize_type,
                    omega=omega, update_direction=update_direction,
                    calc_objective_fn=calc_objective_fn,
                    calc_constraint_fn=calc_constraint_fn,
                    objective_grad=objective_grad, eta=eta, epsilon=epsilon,
                    delta=delta, alpha_fixed=alpha_fixed, alpha_init=alpha_init,
                    decay_factor=decay_factor, armijo_const=armijo_const)
                
                omega = updated_omega.copy()

                if FLAG_SAVE_INNER_STEPS:
                    pi_tmp = softmax(omega)
                    tmp_list.append(env.calc_vpi(pi_tmp, FLAG_RETURN_V_S0=True))
                    
            if FLAG_SAVE_INNER_STEPS:
                vpi_list_inner.append(tmp_list)

    return vpi_list_outer, vpi_list_inner, pi

#----------------------------------------------------------------------
# function for running full experiments
#----------------------------------------------------------------------
def run_experiment(env, pg_method, num_iters, eta, delta, decay_factor, epsilon,
                   FLAG_ANALYTICAL_GRADIENT, num_inner_updates, alpha,
                   FLAG_TRUE_ADVANTAGE, adv_estimate_alg,
                   num_traj_estimate_adv, adv_estimate_stepsize,
                   FLAG_SAVE_INNER_STEPS, zeta, armijo_const,
                   max_backtracking_iters,
                   FLAG_BETA_WARM_START,
                   warm_start_beta_init, warm_start_beta_factor):    
    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct representation)
    theta = init_theta(num_states, num_actions, theta_init=0)
    pi = softmax(theta)
    
    # store the quality of the policies
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
                            armijo_const=armijo_const,
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
