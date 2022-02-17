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

def log_divide(p, q):
    clip_p = np.clip(p, a_min=1e-8, a_max=1-1e-8) 
    clip_q = np.clip(q, a_min=1e-8, a_max=1-1e-8)        
    return np.log(clip_p / clip_q)

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
    assert stepsize_type in ['fixed', 'line_search']

    if stepsize_type == 'fixed':
        if optim_type == 'regularized':
            omega_tmp = omega + alpha_fixed * update_direction
            alpha = alpha_fixed
        elif optim_type == 'constrained':
            raise NotImplementedError()
    elif stepsize_type == 'line_search':
        assert calc_objective_fn is not None
        J_old = calc_objective_fn(omega)
        # the term cm comes from
        # https://en.wikipedia.org/wiki/Backtracking_line_search#Motivation
        cm = armijo_const * (objective_grad * update_direction).sum()
        alpha = alpha_init

        t = 0
        while True:
            omega_tmp = omega + alpha * update_direction
            J_tmp = calc_objective_fn(omega_tmp)
            if calc_constraint_fn is not None:
                C_tmp = calc_constraint_fn(omega_tmp)
                
            if ((optim_type == 'regularized' and J_tmp >= J_old + alpha * cm)
                or
                (optim_type == 'constrained' and # no Armijo condition here
                 J_tmp >= J_old and C_tmp <= delta)):
                break # use this alpha
            elif t > max_backtracking_steps:
                alpha = 1e-16
                omega_tmp = omega + alpha * update_direction
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
# def calc_grad_sPPO(omega, pi_t, adv_t, dpi_t, eta):
#     pi = softmax(omega)
#     grad = dpi_t.reshape(-1, 1) * (pi_t * (adv_t + 1 / eta) - pi / eta)
#     return grad

# def calc_grad_MDPO(omega, pi_t, adv_t, dpi_t, eta):
#     pi = softmax(omega)
#     KL = entropy(pi, pi_t, axis=1).reshape(-1, 1)
#     adv_sum = (pi * adv_t).sum(1).reshape(-1, 1)
#     grad = (dpi_t.reshape(-1, 1) / eta) * pi \
#         * (eta * adv_t - eta * adv_sum - np.log(pi / pi_t) + KL)
#     return grad

#----------------------------------------------------------------------
# Combined Functions
#----------------------------------------------------------------------
def analytical_update_fmapg(pi_old, eta, adv, pg_method):
    assert pg_method in ['sPPO', 'MDPO']
    
    if pg_method == 'sPPO':
        # pi_new = pi_old * np.maximum(1 + eta * adv, 0)  # exact sPPO update
        pi_new = pi_old * np.maximum(1 + eta * adv, 1e-8)
        cnt_neg = np.count_nonzero(1 + eta * adv < 0)
        cnt_neg_adv = np.count_nonzero(adv < 0)
    elif pg_method == 'MDPO':
        pi_new = pi_old * np.exp(eta * adv) # exact MDPO update
        cnt_neg = cnt_neg_adv = None

    pi_new = pi_new / pi_new.sum(1).reshape(-1, 1) # normalize
    return pi_new, cnt_neg, cnt_neg_adv

def calc_objective(omega, pi_t, adv_t, qpi_t, dpi_t, pg_method, epsilon=None):
    pi = softmax(omega)
    if pg_method in ['sPPO']:      
        obj = (dpi_t * (pi_t * adv_t * log_divide(pi, pi_t)).sum(1)).sum()
    elif pg_method in ['TRPO']:
        obj = (dpi_t * (pi * qpi_t).sum(1)).sum()
    elif pg_method in ['MDPO']:
        obj = (dpi_t * (pi * adv_t).sum(1)).sum()
    elif pg_method in ['PPO']:
        raise NotImplementedError()
    return obj

# The KL is essentially KL = (dpi_t * (pi_t * np.log(pi_t / pi)).sum(1)).sum()
# But we use a better way that avoids NaN values from 0/0 division. See
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
def calc_constraint(omega, pi_t, dpi_t, pg_method):
    pi = softmax(omega)
    if pg_method in ['sPPO', 'TRPO']:
        cst = (dpi_t * entropy(pi_t, pi, axis=1)).sum()
    elif pg_method in ['MDPO']:
        cst = (dpi_t * entropy(pi, pi_t, axis=1)).sum()
    return cst

# calculates the grad vector, but stored as an S x A matrix,
def calc_obj_grad(omega, pi_t, adv_t, qpi_t, dpi_t, pg_method, epsilon=None):
    # note that the objective gradients for both TRPO and MDPO are exactly equal
    pi = softmax(omega)    
    if pg_method in ['sPPO']:
        obj_grad = dpi_t.reshape(-1, 1) * pi_t * adv_t
    elif pg_method in ['TRPO']:
        qpi_avg = (pi * qpi_t).sum(1).reshape(-1, 1)
        obj_grad = dpi_t.reshape(-1, 1) * pi * (qpi_t - qpi_avg)
    elif pg_method in ['MDPO']:
        adv_avg = (pi * adv_t).sum(1).reshape(-1, 1)
        obj_grad = dpi_t.reshape(-1, 1) * pi * (adv_t - adv_avg)
    elif pg_method in ['PPO']:
        clipped_pi_t = np.clip(pi_t, a_min=1e-8, a_max=1-1e-8)  
        ratio = pi / clipped_pi_t
        cond = np.logical_or(np.logical_and(adv_t > 0, ratio < 1 + epsilon),
                             np.logical_and(adv_t < 0, ratio > 1 - epsilon))
        cond_adv_avg = (pi * cond * adv_t).sum(1).reshape(-1, 1)
        obj_grad = dpi_t.reshape(-1, 1) * pi * (cond * adv_t - cond_adv_avg)

    return obj_grad
        
def calc_cst_grad(omega, pi_t, dpi_t, pg_method):
    pi = softmax(omega)
    if pg_method in ['sPPO', 'TRPO']:
        cst_grad =  dpi_t.reshape(-1, 1) * (pi - pi_t)
    elif pg_method in ['MDPO']:
        KL = entropy(pi, pi_t, axis=1).reshape(-1, 1)
        cst_grad = dpi_t.reshape(-1, 1) * pi * (log_divide(pi, pi_t) - KL)
    
    return cst_grad

def calc_A_matrix(num_states, num_actions, omega, pi_t, dpi_t, pg_method):
    pi = softmax(omega)
    n = num_states * num_actions
    A = np.zeros((n, n))

    if pg_method in ['TRPO', 'sPPO']:
        for state_i in range(num_states):
            pi_s = pi[state_i, :]
            beg = num_actions * state_i
            end = beg + num_actions
            A[beg:end, beg:end] \
                = dpi_t[state_i] * (np.diag(pi_s) - np.outer(pi_s, pi_s))
    elif pg_method in ['MDPO']:
        for state_i in range(num_states):
            pi_s = pi[state_i, :]
            pi_t_s = pi_t[state_i, :]
            T_vec = pi_s * (log_divide(pi_s, pi_t_s)
                            - entropy(pi_s, pi_t_s) + 1)
            beg = num_actions * state_i
            end = beg + num_actions
            
            # A[beg:end, beg:end] \
            #     = dpi_t[state_i] * (np.diag(T_vec) - 2 * np.outer(pi_s, T_vec)
            #                         + np.outer(pi_s, pi_s))
            A[beg:end, beg:end] \
                = dpi_t[state_i] * (np.diag(T_vec) - np.outer(pi_s, T_vec)
                                    - np.outer(T_vec, pi_s)
                                    + np.outer(pi_s, pi_s))
    else:
        raise NotImplementedError()
    
    return A

# calc_A_matrix_slow is an alternate way of calculating the above matrix, and
# serves as a check for the function calc_A_matrix()
# def calc_A_matrix_slow(num_states, num_actions, omega, pi_t, dpi_t, pg_method):
#     # only for verifying if calc_A_matrix() works correctly
#     pi = softmax(omega)
#     n = num_states * num_actions
#     A = np.zeros((n, n))

#     if pg_method in ['TRPO', 'sPPO']:
#         for state_i in range(num_states):
#             for action_i in range(num_actions):
#                 for state_j in range(num_states):
#                     for action_j in range(num_actions):
#                         row = state_i * num_actions + action_i
#                         col = state_j * num_actions + action_j
#                         A[row, col] = (state_i == state_j) * dpi_t[state_j] \
#                             * ((action_i == action_j) - pi[state_j, action_i]) \
#                             * pi[state_j, action_j]
#     elif pg_method in ['MDPO']:
#         for state_i in range(num_states):
#             for action_i in range(num_actions):
#                 for state_j in range(num_states):
#                     for action_j in range(num_actions):
#                         row = state_i * num_actions + action_i
#                         col = state_j * num_actions + action_j

#                         clipped_pi_t = np.clip(pi_t[state_i, action_j],
#                                                  a_min=1e-8, a_max=1-1e-8)
#                         T_itd_s_aprime = \
#                             np.log(pi[state_i, action_j] / clipped_pi_t) \
#                             - entropy(pi[state_i, :], pi_t[state_i, :])
                        
#                         A[row, col] = (state_i == state_j) * dpi_t[state_j] \
#                             * ((action_i == action_j) * pi[state_i, action_i] \
#                                * T_itd_s_aprime - 2 * pi[state_i, action_j] \
#                                * pi[state_i, action_i] * T_itd_s_aprime \
#                                + pi[state_i, action_j] * pi[state_i, action_i])
#     else:
#         raise NotImplementedError() 
#     return A

# compute the maximum stepsize using the equations given in Appendix C of
# Schulman et al. (2015)
def calc_max_stepsize(update_direction, A, delta, alpha_max):
    if np.linalg.norm(update_direction) < 1e-128:
        alpha = alpha_max
    else:
        update_direction_flatten = update_direction.reshape(-1, 1)
        denom = np.matmul(np.matmul(update_direction_flatten.T, A),
                          update_direction_flatten).item()
        if denom < 1e-128:
            alpha = alpha_max
        else:
            alpha = 2 * delta / denom
            alpha = np.sqrt(alpha)
            
    if alpha > alpha_max:
        alpha = alpha_max
        
    return alpha

#----------------------------------------------------------------------
# function for running full experiments
#----------------------------------------------------------------------
def run_experiment(env, pg_method, num_outer_loop, num_inner_loop,
                   FLAG_SAVE_INNER_STEPS,
                   alpha_max, FLAG_WARM_START, warm_start_factor,
                   max_backtracking_steps,
                   optim_type, stepsize_type, eta, epsilon, delta,
                   alpha_fixed, decay_factor, armijo_const):
    assert pg_method in ['sPPO', 'MDPO', 'PPO', 'TRPO']
    assert optim_type in ['regularized', 'constrained', 'analytical']
    assert stepsize_type in ['fixed', 'line_search']

    num_states = env.state_space
    num_actions = env.action_space

    # initialize pi (uniform random tabular policy with direct rep)
    theta = np.zeros((num_states, num_actions))

    # theta[:, 1] = 5
    # theta[0, 1] = 0
    # theta[0, 3] = 5
    # theta[9, 1] = 0
    # theta[9, 2] = 5
    
    pi = softmax(theta)
    
    # store the quality of the policies
    vpi_outer_list = []
    grad_norm_list = []
    grad_jpi_outer_list = []
    vpi_inner_list = [] if FLAG_SAVE_INNER_STEPS else None
    alpha_used_list = [] if FLAG_SAVE_INNER_STEPS else None
    grad_lpi_inner_list = []

    # store how many times does (1 + eta * adv) < 0 for analytical sPPO update
    if pg_method == 'sPPO' and optim_type == 'analytical':
        cnt_neg_list = []
        cnt_neg_adv_list = []
    else:
        cnt_neg_list = cnt_neg_adv_list = None
        
    # outer learning loop
    for T in range(num_outer_loop):
        dpi = env.calc_dpi(pi)
        qpi = env.calc_qpi(pi)
        adv = qpi - env.calc_vpi(pi).reshape(-1, 1)

        J_pi_grad = dpi.reshape(-1, 1) * pi * adv
        grad_jpi_outer_list.append(np.linalg.norm(J_pi_grad))

        # if np.linalg.norm(J_pi_grad) < 1e-16:
        #     print('kill outside')
        #     break
        
        print(T, np.linalg.norm(J_pi_grad))

        vpi_outer_list.append(env.calc_vpi(pi, FLAG_RETURN_V_S0=True))

        if optim_type == 'analytical': 
            pi, cnt_neg, cnt_neg_adv = analytical_update_fmapg(
                pi_old=pi, eta=eta, adv=adv, pg_method=pg_method)
            if pg_method == 'sPPO':
                cnt_neg_list.append(cnt_neg)
                cnt_neg_adv_list.append(cnt_neg_adv)
        else:
            # multiple gradient based updates
            if FLAG_SAVE_INNER_STEPS:
                vpi_tmp_list = [env.calc_vpi(pi, FLAG_RETURN_V_S0=True)]
                alpha_tmp_list = []
                grad_lpi_inner_tmp_list = []

            omega = theta.copy()
            used_alpha = None
                            
            # inner learning loop
            for k in range(num_inner_loop): # do one grad ascent step
                #--------------------------------------------------------
                # Calculate the objective_grad and the update_direction
                #--------------------------------------------------------
                if optim_type == 'regularized':
                    obj_grad = calc_obj_grad(omega=omega, pi_t=pi, adv_t=adv,
                                             qpi_t=qpi, dpi_t=dpi,
                                             pg_method=pg_method,
                                             epsilon=epsilon)
                    if pg_method in ['PPO']:
                        update_direction = obj_grad
                    else:
                        cst_grad = calc_cst_grad(omega=omega, pi_t=pi, dpi_t=dpi,
                                                 pg_method=pg_method)
                        update_direction = obj_grad - (1 / eta) * cst_grad
                    objective_grad = update_direction
                elif optim_type == 'constrained':
                    # calc update_direction = A^{-1} grad.
                    # We first flatten grad, calculate update_direction using
                    # the above equation, and then reshape it again into an
                    # S x A matrix
                    objective_grad = calc_obj_grad(
                        omega=omega, pi_t=pi, adv_t=adv, qpi_t=qpi, dpi_t=dpi,
                        pg_method=pg_method, epsilon=epsilon)
                    objective_grad_flatten = objective_grad.reshape(-1, 1)
                    A = calc_A_matrix(num_states=num_states,
                                      num_actions=num_actions, omega=omega,
                                      pi_t=pi, dpi_t=dpi, pg_method=pg_method)
                    # A_slow = calc_A_matrix(
                    #     num_states=num_states, num_actions=num_actions,
                    #     omega=omega, pi_t=pi, dpi_t=dpi, pg_method=pg_method)
                    
                    # s_flatten = np.linalg.solve(A, objective_grad_flatten)
                    # had problems with A being singular; so used pseudo-inverse
                    update_direction_flatten = np.matmul(
                        np.linalg.pinv(A), objective_grad_flatten)
                    update_direction = update_direction_flatten.reshape(
                        objective_grad.shape)

                #--------------------------------------------------------
                # Calculate objective_fn, constraint_fn, and alpha_init
                #--------------------------------------------------------
                if stepsize_type == 'fixed':
                    calc_objective_fn = None
                    calc_constraint_fn = None
                    alpha_init = None
                elif stepsize_type == 'line_search':
                    calc_objective_fn = lambda omega: calc_objective(
                        omega=omega, pi_t=pi, adv_t=adv, qpi_t=qpi, dpi_t=dpi,
                        pg_method=pg_method, epsilon=epsilon)
                    calc_constraint_fn = lambda omega: calc_constraint(
                        omega=omega, pi_t=pi, dpi_t=dpi, pg_method=pg_method)

                    if optim_type == 'constrained':
                        alpha_init = calc_max_stepsize(
                            update_direction=update_direction, A=A, delta=delta,
                            alpha_max=alpha_max)
                    elif FLAG_WARM_START and used_alpha is not None:
                        alpha_init = warm_start_factor * used_alpha
                    else:
                        alpha_init = alpha_max

                if np.linalg.norm(objective_grad) < 1e-128:
                    updated_omega = omega.copy()
                    used_alpha = 0
                    print('kill inside')
                    break
                else: # make a single inner loop update
                    updated_omega, used_alpha = make_single_inner_loop_update(
                        optim_type=optim_type, stepsize_type=stepsize_type,
                        omega=omega, update_direction=update_direction,
                        calc_objective_fn=calc_objective_fn,
                        calc_constraint_fn=calc_constraint_fn,
                        objective_grad=objective_grad, eta=eta,
                        epsilon=epsilon, delta=delta, alpha_fixed=alpha_fixed,
                        alpha_init=alpha_init, decay_factor=decay_factor,
                        armijo_const=armijo_const,
                        max_backtracking_steps=max_backtracking_steps)
                
                omega = updated_omega.copy()

                if FLAG_SAVE_INNER_STEPS:
                    pi_tmp = softmax(omega)
                    vpi_tmp_list.append(
                        env.calc_vpi(pi_tmp, FLAG_RETURN_V_S0=True))
                    if k % 10 == 0:
                        grad_lpi_inner_tmp_list.append(
                            np.linalg.norm(objective_grad))
                    alpha_tmp_list.append(used_alpha)

            if FLAG_SAVE_INNER_STEPS:
                vpi_inner_list.append(vpi_tmp_list)
                alpha_used_list.append(alpha_tmp_list)
                grad_lpi_inner_list.append(grad_lpi_inner_tmp_list)

            # update the policy to the new approximate point
            theta = omega.copy()
            pi = softmax(theta)

    dat = dict()
    dat['vpi_outer_list'] = vpi_outer_list
    dat['grad_jpi_outer_list'] = grad_jpi_outer_list
    dat['final_theta'] = theta.tolist()

    dat['cnt_neg_list'] = cnt_neg_list
    dat['cnt_neg_adv_list'] = cnt_neg_adv_list
    
    dat['vpi_inner_list'] = vpi_inner_list
    dat['grad_lpi_inner_list'] = grad_lpi_inner_list
    dat['alpha_used_list'] = alpha_used_list
    
    return dat
