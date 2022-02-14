import numpy as np
import copy
import pdb

#======================================================================
# Saving the environment data from Simone's environment
#----------------------------------------------------------------------
# from src.envs.gridworld_mdp import cliff_gw
# env = cliff_gw(gamma=0.99)
# np.savez('cliff_world_env', P=env.P, r=env.R, mu=env.p0,
#          terminal_states=env.terminal_states)
# arr_dict = np.load('cliff_world_env.npz')
# P_simone = arr_dict['P']
# r_simone = arr_dict['r']
# mu_simone = arr_dict['mu']
# terminal_states_simone = [48]
#======================================================================

#======================================================================
# Simplified CliffWorld (re-write)
#----------------------------------------------------------------------
# -------------------         4 is the goal state
# | 4 | 9 | 14 | 19 |         20 is terminal state reached only via the state 4
# -------------------         3, 2, 1 are chasms  
# | 3 | 8 | 13 | 18 |         0 is the start state
# ------------------- 
# | 2 | 7 | 12 | 17 |         all transitions are deterministic
# -------------------         Actions: 0=down, 1=up, 2=left, 3=right
# | 1 | 6 | 11 | 16 |
# ------------------------    rewards are all zeros except at chasms (-100)
# | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
# ------------------------
#----------------------------------------------------------------------

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

P_CliffWorld = copy.deepcopy(P)
r_CliffWorld = copy.deepcopy(r)
mu_CliffWorld = copy.deepcopy(mu)
terminal_states_CliffWorld = copy.deepcopy(terminal_states)
#======================================================================

#======================================================================
# Deep Sea Treasure
#----------------------------------------------------------------------
# ------------------------    20 is the goal state
# | 4 | 9 | 14 | 19 | 24 |    0, 5, 10, 15, 20 are terminal states
# ------------------------    4 is the start state
# | 3 | 8 | 13 | 18 | 23 |    
# ------------------------
# | 2 | 7 | 12 | 17 | 22 |    all transitions are deterministic
# ------------------------    Actions: 0=left, 1=right
# | 1 | 6 | 11 | 16 | 21 |
# ------------------------    rewards are all -0.01/5 except at state 20
# | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
# ------------------------
#----------------------------------------------------------------------
terminal_states = [0, 5, 10, 15, 20]

# environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
P = np.zeros((25, 2, 25))
for state_idx in range(25):
    for action_idx in range(2):
        if state_idx in terminal_states: # terminal states
            new_state_idx = state_idx
        else: # move according to the deterministic dynamics
            x_new = x_old = state_idx // 5
            y_new = y_old = state_idx % 5
            if action_idx == 0: # left
                x_new = np.clip(x_old - 1, 0, 4)
            elif action_idx == 1: # right
                x_new = np.clip(x_old + 1, 0, 4)
            y_new = y_old - 1
            new_state_idx = 5 * x_new + y_new

        P[state_idx, action_idx, new_state_idx] = 1
        
r = (-0.01 / 5) * np.ones((25, 2))
r[16, 1] = r[21, 1] = +1 # positive reward for finding the goal terminal state
for s in terminal_states:
    r[s, :] = 0

mu = np.zeros(25)
mu[4] = 1

P_DeepSeaTreasure = copy.deepcopy(P)
r_DeepSeaTreasure = copy.deepcopy(r)
mu_DeepSeaTreasure = copy.deepcopy(mu)
terminal_states_DeepSeaTreasure = copy.deepcopy(terminal_states)

#======================================================================
        
class TabularMDP():
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
