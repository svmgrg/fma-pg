import numpy as np
import pdb
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy

#======================================================================
# Plotting functions
#======================================================================
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


#======================================================================
# Environment
#======================================================================
# environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
P = np.zeros((21, 4, 21))
for state_idx in range(21):
    for action_idx in range(4):
        if state_idx in [1, 2, 3]: # chasms: reset to start state 0
            new_state_idx = 0
        elif state_idx == 4: # goal state: agent always goes to 20
            new_state_idx = 20
        elif state_idx == 20: # terminal state
            new_state_idx = 20
        else: # move according to the deterministic dynamics
            x_new = x_old = state_idx // 5
            y_new = y_old = state_idx % 5
            if action_idx == 0: # Down
                y_new = np.clip(y_old - 1, 0, 4)
            elif action_idx == 1: # Up
                y_new = np.clip(y_old + 1, 0, 4)
            elif action_idx == 2: # Left
                x_new = np.clip(x_old - 1, 0, 3)
            elif action_idx == 3: # Right
                x_new = np.clip(x_old + 1, 0, 3)
            new_state_idx = 5 * x_new + y_new

        P[state_idx, action_idx, new_state_idx] = 1
        
r = np.zeros((21, 4))
r[1, :] = r[2, :] = r[3, :] = -100 # negative reward for falling into chasms
r[4, :] = +1 # positive reward for finding the goal terminal state

mu = np.zeros(21)
mu[0] = 1

terminal_states = [20]
#======================================================================
# Simplified CliffWorld (re-write)
#----------------------------------------------------------------------
# -------------------         4 is the goal state
# | 4 | 9 | 14 | 19 |         20 is terminal state reached only via the state 4
# -------------------         3, 2, 1 are chasms  
# | 3 | 8 | 13 | 18 |         0 is the start state
# ------------------- 
# | 2 | 7 | 12 | 17 |         all transitions are determinitic
# -------------------         Actions: 0=down, 1=up, 2=left, 3=right
# | 1 | 6 | 11 | 16 |
# ------------------------    rewards are all zeros except at chasms (-100)
# | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
# ------------------------
#----------------------------------------------------------------------

        
class CliffWorld():
    def __init__(self, P, r, mu, terminal_states, gamma, episode_cutoff_length,
                 reward_noise):
        self.P = P
        self.r = r
        self.mu = mu
        self.terminal_states = terminal_states
        
        self.state_space = self.r.shape[0]
        self.action_space = self.r.shape[1]

        self.gamma = gamma
        self.episode_cutoff_length = episode_cutoff_length
        self.reward_noise = reward_noise

        self.t = None
        self.state = None

    def reset(self):
        self.t = 0
        self.state = np.random.choice(a=self.state_space, p=self.mu)
        return self.state

    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.action_space)

        reward = self.r[self.state, action] \
            + np.random.normal(loc=0, scale=self.reward_noise)
        self.state = np.random.choice(a=self.state_space,
                                      p=self.P[self.state, action])
        self.t = self.t + 1

        done = 'false'
        if self.state in self.terminal_states:
            done = 'terminal'
        elif self.t > self.episode_cutoff_length:
            done = 'cutoff'

        return self.state, reward, done

    # for computing v^pi = (I - gamma P_pi)^{-1} r_pi
    # environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
    # policy pi: (i, j)th element = Pr{A = a_j | S = s_i}
    # P_pi(s' | s) = sum_a P(s' | s, a) * pi(a | s)
    def calc_vpi(self, pi, FLAG_RETURN_V_S0=False):
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        r_pi = np.einsum('xa,xa->x', self.r, pi)
        v_pi = np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi,
                               r_pi)
        if FLAG_RETURN_V_S0: # calculate v_s0
            v_s0 = np.dot(self.mu, v_pi)
            return v_s0
        else:
            return v_pi

    # compute q^pi = r(s, a) + gamma * sum_s' p(s' | s, a) * v^pi(s')
    def calc_qpi(self, pi):
        v_pi = self.calc_vpi(pi)
        q_pi = self.r + self.gamma * np.einsum('xay,y->xa', self.P, v_pi)
        return q_pi

    # computing the normalized occupancy measure
    # d^pi = (1 - gamma) * mu (I - gamma P_pi)^{-1};   mu = start state dist
    def calc_dpi(self, pi):
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        d_pi = (1 - self.gamma) * \
            np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi.T,
                            self.mu)
        d_pi /= d_pi.sum() # for addressing numerical errors
        return d_pi

    def calc_q_star(self, num_iters=1000):
        q = np.zeros((self.state_space, self.action_space))
        for i in range(num_iters):
            q_new = self.r + np.einsum("xay,y->xa",
                                       self.gamma * self.P, q.max(1))
            q = q_new.copy()
        return q

    def calc_v_star(self, num_iters=1000):
        v = np.zeros(self.state_space)
        for i in range(num_iters):
            v_new = self.r + np.einsum("xay,y->xa", self.gamma * self.P, v)
            v_new = v_new.max(1)
            v = v_new.copy()
        return v

    def calc_pi_star(self, num_iters=1000): # just go greedy wrt q_star
        q_star = self.calc_q_star(num_iters=num_iters)
        pi_star = np.zeros((self.state_space, self.action_space))
        pi_star[range(self.state_space), q_star.argmax(1)] = 1
        return pi_star
#======================================================================

gamma = 0.9
episode_cutoff_length = None
reward_noise = None
env = CliffWorld(P=P, r=r, mu=mu, terminal_states=terminal_states,
                 gamma=gamma, episode_cutoff_length=episode_cutoff_length,
                 reward_noise=reward_noise)

def softmax(x):
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    out = e_x / e_x.sum(1).reshape(-1, 1)
    return out

#======================================================================
# Learning Curves
#======================================================================

num_actions = env.action_space
num_states = env.state_space
start_state = 0

num_runs = 1
num_steps = 1000
policy_stepsize = 0.1
entropy_reg_strength = 0.7 # not sure if the entropy code is correct though

policy_features = np.vstack((np.eye(20), np.zeros(20)))
policy_weight = 0 * np.ones((policy_features.shape[1], num_actions))

# policy_weight[:, 1] = 5
# policy_weight[0, 1] = 0
# policy_weight[0, 3] = 5
# policy_weight[9, 1] = 0
# policy_weight[9, 2] = 5

theta = np.matmul(policy_features, policy_weight)
pi = softmax(theta)
d_gamma = env.calc_dpi(pi)
q_pi = env.calc_qpi(pi)
v_pi = env.calc_vpi(pi)

perf = [v_pi[start_state]]

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
plot_grid(axs[0])
plot_policy(axs[0], pi)

tic = time.time()
for i in range(num_steps):
    gradJ = np.zeros(policy_weight.shape)
    gradH = np.zeros(policy_weight.shape)
    for s in range(num_states):
        feat_s = policy_features[s].reshape(-1, 1)

        gradJ_s = np.matmul(
            feat_s, (pi[s] * (q_pi[s] - v_pi[s])).reshape(1, -1))
        gradJ += d_gamma[s] * gradJ_s

        gradH_s = np.matmul(
            feat_s, (pi[s] * (np.log(pi[s]) + entropy(pi[s]))).reshape(1, -1))
        gradH += d_gamma[s] * gradH_s

    grad = gradJ - entropy_reg_strength * gradH
    policy_weight += policy_stepsize * grad

    theta = np.matmul(policy_features, policy_weight)
    pi = softmax(theta)
    d_gamma = env.calc_dpi(pi)
    q_pi = env.calc_qpi(pi)
    v_pi = env.calc_vpi(pi)

    perf.append(v_pi[start_state])

print('Time taken:', time.time() - tic)    
    
# axs[1].set_ylim([-1, 0.6])
axs[1].plot(perf)

plot_grid(axs[2])
plot_policy(axs[2], pi)

plt.show()

exit()


#======================================================================
# Stepsize sensitivity
#======================================================================
num_actions = env.action_space
num_states = env.state_space
start_state = 0

num_runs = 1
num_steps = 10
policy_stepsize_list = np.arange(0, 100, 1)

sensitivity_perf = []
tic = time.time()
for policy_stepsize in policy_stepsize_list:
    policy_features = np.vstack((np.eye(20), np.zeros(20)))
    policy_weight = 0 * np.ones((policy_features.shape[1], num_actions))

    policy_weight[:, 1] = 5
    policy_weight[0, 1] = 0
    policy_weight[0, 3] = 5
    policy_weight[5, 1] = 0
    policy_weight[5, 3] = 5
    policy_weight[9, 1] = 0
    policy_weight[9, 2] = 5

    theta = np.matmul(policy_features, policy_weight)
    pi = softmax(theta)
    d_gamma = env.calc_dpi(pi)
    q_pi = env.calc_qpi(pi)
    v_pi = env.calc_vpi(pi)

    perf = [v_pi[start_state]]

    for i in range(num_steps):
        grad = np.zeros(policy_weight.shape)
        for s in range(num_states):
            feat_s = policy_features[s].reshape(-1, 1)
            grad_s = np.matmul(feat_s,
                               (pi[s] * (q_pi[s] - v_pi[s])).reshape(1, -1))
            grad += d_gamma[s] * grad_s

        policy_weight += policy_stepsize * grad

        theta = np.matmul(policy_features, policy_weight)
        pi = softmax(theta)
        d_gamma = env.calc_dpi(pi)
        q_pi = env.calc_qpi(pi)
        v_pi = env.calc_vpi(pi)

        perf.append(v_pi[start_state])

    perf = np.array(perf)
    sensitivity_perf.append(perf[-10].mean())

    
print('Time taken:', time.time() - tic)    

plt.plot(sensitivity_perf)

plt.show()
exit()
